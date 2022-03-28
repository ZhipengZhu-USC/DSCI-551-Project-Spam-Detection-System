from flask import Flask, render_template, request, flash, redirect, url_for
from firebase import *
import pandas as pd
import os,pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import requests
import json


# import my functions
from text_process import *

app = Flask(__name__)
app.secret_key = "123"

@app.route("/hello")
def home():
    return render_template("index.html")

@app.route("/data")
def data():
    return render_template("data.html", file_list = get_file_list("raw_data/") )

@app.route("/upload", methods= ['GET','POST'])
def data_upload():
    return render_template("upload.html")

@app.route("/upload_file", methods= ['GET','POST'])
def upload_result():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
            upload(uploaded_file.filename,"raw_data/")
            #os.remove(uploaded_file.filename)

        return "upload successful"


@app.route("/ML")
def ML_list():
    return render_template("ML.html", file_list = get_file_list("ML/") )

@app.route("/viewFile", methods= ['GET','POST'])
def viewFile():
    if request.method == 'POST':
        name = request.form["submit_button"]
        name = name[5:]
        download(name,"raw_data/")
        df = pd.read_csv(name)
        #metadata of the file
        rows = df.shape[0]
        columns=df.shape[1]
        attributes=""
        file_size=os.path.getsize(name)
        size = round(file_size / 1024, 0)

        attributes = ", ".join(df.columns)
        #df.to_html('templates/' + name[:-4] + '.html')
       # os.remove(name)

        file_size = os.path.getsize('spam_ham_dataset.csv')
        print("File Size is :", round(file_size / 1024, 0), "KB")
        #return render_template(name[:-4] + '.html',rows=rows, columns=columns, attributes=attributes, size=size)

        return render_template('myFile.html',table=df.to_html(classes='data',index=False,justify='left'), title = "Metadata",rows=rows, columns=columns, attributes=attributes, size=size)

@app.route("/viewFeature")
def viewFeature():
    r = requests.get("https://spam-email-classifier-9156c-default-rtdb.firebaseio.com/extracted_features/ham.json")
    mydict = r.json()
    print(mydict)
    mylist = []
    for key in mydict:
        mylist.append([mydict[key]['word'], mydict[key]['frequency']])
    df_ham = pd.DataFrame(mylist, columns=["word", "frequency"])

    r = requests.get("https://spam-email-classifier-9156c-default-rtdb.firebaseio.com/extracted_features/spam.json")
    mydict = r.json()
    print(mydict)
    mylist = []
    for key in mydict:
        mylist.append([mydict[key]['word'], mydict[key]['frequency']])
    df_spam = pd.DataFrame(mylist, columns=["word", "frequency"])

    print(df_ham)
    print(df_spam)
    return render_template("feature.html", tables=[df_ham.sort_values(by='frequency',ascending=False).to_html(classes='data', index=False),df_spam.sort_values(by='frequency',ascending= False).to_html(classes='data', index=False)],
                           titles=["Most frequent words in non-spam","Most frequent words in spam"])


@app.route("/viewModel", methods= ['GET','POST'])
def viewModel():
    #model_name=get_name()
    download("MNB_model.mod","ML/")
    download("train_report.dict", "ML/")
    download("test_report.dict", "ML/")
    download("vect.vec", "ML/")
    # load vectorizer
    with open('vect.vec', 'rb') as f:
        vectorizer = pickle.load(f)
    # load model
    with open('MNB_model.mod', 'rb') as f:
        NBclassifier = pickle.load(f)
    # load reports
    with open('train_report.dict', 'rb') as f:
        train_report = pickle.load(f)
    with open('test_report.dict', 'rb') as f:
        test_report = pickle.load(f)

    df_train_report = pd.DataFrame(train_report)
    df_test_report = pd.DataFrame(test_report)

    return render_template('viewModel.html', tables=[df_train_report.to_html(classes='data',index=False),df_test_report.to_html(classes='data',index=False)], titles=["train","test"])



"""
#test the post method
@app.route("/test", methods = ['GET','POST'])
def test():
    if request.method == 'POST':
        name = request.form['nm']
        download(name, "raw_data/")
        df = pd.read_csv(name)
        df.to_html('templates/'+name[:-4]+'.html')
       # os.remove(name)
        return render_template(name[:-4]+'.html')
"""



@app.route("/train",methods = ['GET','POST'])
def train():
    name = request.form["train_model"]
    name = name[38:]
    download(name, "raw_data/")
    dataframe = pd.read_csv(name)

    x_raw = dataframe["text"]
    y_raw = dataframe["label"]

    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=0.20, random_state=0)

    vectorizer = CountVectorizer(analyzer=process_text)
    bow_train = vectorizer.fit_transform(x_train)

    # store vectorizer
    with open('vect.vec', 'wb') as m:
        pickle.dump(vectorizer, m)

    upload("vect.vec","ML/")
    # vocab
    # print("vocab: ",vectorizer.get_feature_names())

    # format bow
    bow_train_df = pd.DataFrame(bow_train.toarray(), columns=vectorizer.get_feature_names())
    bow_train_array = bow_train.toarray()

    # step 4: build models

    # naive bayes model
    NBclassifier = MultinomialNB().fit(bow_train, y_train)


    # step 5: evaluate model

    print("training MNB")

    train_pred = NBclassifier.predict(bow_train)
    train_report = classification_report(y_train, train_pred, output_dict=True)
    train_report_filtered = {}
    train_report_filtered['precision'] = [round(train_report['spam']['precision'], 3)]
    train_report_filtered['recall'] = [round(train_report['spam']['recall'], 3)]
    train_report_filtered['f1-score'] = [round(train_report['spam']['f1-score'], 3)]
    train_report_filtered['accuracy'] = [round(train_report['accuracy'], 3)]
    #df_report = pd.DataFrame(report).transpose()
    with open('train_report.dict', 'wb') as m:
        pickle.dump(train_report_filtered, m)
    upload("train_report.dict","ML/")

    with open('MNB_model.mod', 'wb') as m:
        pickle.dump(NBclassifier, m)
    upload("MNB_model.mod","ML/")

    print("test MNB")  # here need to create new bow for test data, using the TRAINING vocabulary!!! AKA use the same vectorizer
    bow_test = vectorizer.transform(x_test)
    test_pred = NBclassifier.predict(bow_test)
    test_report = (classification_report(y_test, test_pred, output_dict=True))
    test_report_filtered = {}
    test_report_filtered['precision'] = [round(test_report['spam']['precision'],3)]
    test_report_filtered['recall'] = [round(test_report['spam']['recall'],3)]
    test_report_filtered['f1-score'] = [round(test_report['spam']['f1-score'],3)]
    test_report_filtered['accuracy'] = [round(test_report['accuracy'],3)]
    print(test_report_filtered)


    with open('test_report.dict', 'wb') as m:
        pickle.dump(test_report_filtered, m)
    upload("test_report.dict","ML/")
    #calculate top words
    #ham
    bow_train_df['label'] = y_train.tolist()
    bow_train_ham = bow_train_df.loc[bow_train_df['label'] == 'ham']
    bow_train_ham = bow_train_ham.drop('label', 1)

    word_sum_h = bow_train_ham.sum()
    sum_h = list(word_sum_h)

    words_h = list(word_sum_h.index)
    Series_Dict = {"word": words_h, "frequency": sum_h}
    Agg_DF_h = pd.DataFrame(Series_Dict).sort_values(by=['frequency'], ascending=False)

    # calculate top words
    # spam
    bow_train_df['label'] = y_train.tolist()
    bow_train_spam = bow_train_df.loc[bow_train_df['label'] == 'spam']
    bow_train_spam = bow_train_spam.drop('label', 1)

    word_sum_s = bow_train_spam.sum()
    sum_s = list(word_sum_s)

    words_s = list(word_sum_s.index)
    Series_Dict = {"word": words_s, "frequency": sum_s}
    Agg_DF_s = pd.DataFrame(Series_Dict).sort_values(by=['frequency'], ascending=False)

    #upload features to RTDB
    dict_h = Agg_DF_h.head(20).to_dict("index")
    json_h = json.dumps(dict_h)
    r = requests.put("https://spam-email-classifier-9156c-default-rtdb.firebaseio.com/extracted_features/ham.json",
                      data=json_h)
    dict_s = Agg_DF_s.head(20).to_dict("index")
    json_s = json.dumps(dict_s)
    r = requests.put("https://spam-email-classifier-9156c-default-rtdb.firebaseio.com/extracted_features/spam.json",
                        data=json_s)

    df_train_report = pd.DataFrame.from_dict(train_report_filtered)
    df_test_report = pd.DataFrame.from_dict(test_report_filtered)

    return render_template('train_result.html',tables=[Agg_DF_h.head(20).to_html(classes='data',index=False), Agg_DF_s.head(20).to_html(classes='data',index=False),df_train_report.to_html(classes='data',index=False), df_test_report.to_html(classes='data',index=False)],titles=["most frequent words in non-spam","most frequent words in spam","train", "test"])



@app.route("/predict", methods= ['GET','POST'])
def predict():
    if request.method == 'POST':
        content = request.form["nm"]
        download("MNB_model.mod", "ML/")
        download("vect.vec", "ML/")
        # load vectorizer
        with open('vect.vec', 'rb') as f:
            vectorizer = pickle.load(f)
        # load model
        with open('MNB_model.mod', 'rb') as f:
            NBclassifier = pickle.load(f)

        bow_vector = vectorizer.transform([content])
        test_pred = NBclassifier.predict(bow_vector)
        # print(classification_report(y_test, test_pred))

        #upload results to firebase RTDB

        data = [[content, test_pred[0]]]
        mydf = pd.DataFrame(data, columns = ['text', 'prediction'])
        mydict = mydf.to_dict("index")
        json_email = json.dumps(mydict)
        r = requests.post("https://spam-email-classifier-9156c-default-rtdb.firebaseio.com/inference_results.json", data=json_email)

        return test_pred[0]
    return render_template("predict.html")

@app.route("/prediction_history")
def history():
    r = requests.get("https://spam-email-classifier-9156c-default-rtdb.firebaseio.com/inference_results.json")
    mydict = r.json()
    mylist =[]
    for key in mydict:
        mylist.append([mydict[key][0]['text'],mydict[key][0]['prediction']])
    df_history=pd.DataFrame(mylist, columns=["text","prediction"])
    return render_template("prediction_history.html",table = df_history.to_html(classes='data'), title = "history of predictions" )


