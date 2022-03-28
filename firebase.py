import pyrebase
config = {
        "apiKey": "AIzaSyBQCDU0eZgx4oTnjXrKVINzOBLneRYyq7k",
        "authDomain": "spam-email-classifier-9156c.firebaseapp.com",
        "databaseURL": "https://spam-email-classifier-9156c-default-rtdb.firebaseio.com",
        "projectId": "spam-email-classifier-9156c",
        "storageBucket": "spam-email-classifier-9156c.appspot.com",
        "messagingSenderId": "208213134148",
        "appId": "1:208213134148:web:f4d66727d275ab94dca511",
        "serviceAccount": "serviceAccountKey.json"
    }

def get_file_list(folder):
    # Your web app's Firebase configuration
    to_return=[]
    firebase_storage = pyrebase.initialize_app(config)
    storage = firebase_storage.storage()
    all_files = storage.list_files()
    for file in all_files:
        if file.name[:len(folder)] == folder:
            to_return.append(file.name[len(folder):])
    return to_return

def download(filename, folder):

    firebase_storage = pyrebase.initialize_app(config)
    storage = firebase_storage.storage()
    path_on_cloud = folder + filename
    storage.child(path_on_cloud).download(filename)

def upload(filename, folder):
    firebase_storage = pyrebase.initialize_app(config)
    storage = firebase_storage.storage()
    path_on_cloud=folder+filename
    storage.child(path_on_cloud).put(filename)

