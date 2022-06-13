## How to run this app locally:

### Setup/Requirements:
* Text Editor or IDE (I personally recommend vscode)
* Browser (I personally recommend Chrome)
* Python (required)
* install all libraries contained in requirement.txt

### File Structure:

```
project (main project folder)
│   Dockerfile
│   main.py
│   requirements.txt 
│
└───ml-model
│   │    
│   └───Model
│     	  my_model1.h5
│  
└───backup-code (codes that are not used because they are not in accordance with the project)
│   	  app-v1.py
│   	  app.py
│   	  app3.py
│   	  coba.py
│          
└───testing
│   	  test.py (used to test the API locally or can also use postman and similar apps)
       
```

**Note:** you may need to change some file paths to match your system path structure. 

## Steps:
* Download or clone the files above
* make sure the file path is correct
* Open and run the main.py file in terminal using *flask run* 
* Open test.py and change the address to "localhost:5000". 
* Select an image file using image directory in the open function
* run test.py in the terminal, and then wait for the response
* Welldone!!!

## How to deploy this app in cloud run

### 1. Setup Google Cloud 
- Select existing or Create new project
- Enable Cloud Run API and Cloud Build API

### 2. Install and init Google Cloud SDK
- https://cloud.google.com/sdk/docs/install

### 3. Dockerfile, requirements.txt, .dockerignore
- can be seen in the main folder / root folder, for complete documentation can be seen on the following page:https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service

### 4. Build the Code and Deploy to Cloud
```
gcloud builds submit --tag gcr.io/<project_id>/<func_name> // in this case is upload_file()
gcloud run deploy --image gcr.io/<project_id>/<func_name> --platform managed
```

### Test
- Last is test the code with `test.py` if app run normaly the response will be the prediction result


### Other Repo for this project: https://github.com/amfajar/sampahku/tree/master