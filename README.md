#### Music genre classifier

In the lectures I had the chance to interact with Tensorflow and use a pre-trained model to classify clothes using images

To practice further I wanted to apply the same approach but instead of clothes use audio samples of different music genres
and try to classify them. While I was searching for a dataset I found the GTZAN dataset which has up to 1000 audio samples
of 10 different music genres, so I only had to figure out what could be used as an image using the audio samples, so I
chose the mel spectrogram which is a visual representation of the frequency content of an audio signal over time

The objective is to classify an audio sample of a music genre from the ones in the GTZAN dataset using a keras pre-trained
model and mel spectrograms of the audio samples as input. The process mainly focuses on Transfer Learning and searching for
tuned parameters for Pooling, Dense and Inner Layer, Optimization, Droput and Data Augmentation, using extracted features
from a pre-trained model in this case MobileNetV2 and adjust each layer as learned in the lectures

Once the model is tuned it has to be exported to onnx format, deployed locally using FastAPI and later be packed into a Docker
Image in order to be used with kubernetes as shown in previous lectures and workshops (that is what I am aiming for)

This time I am not deploying it to the cloud but I am going to cover the steps to deploy it locally with kubernetes which covers
most of the workshops and lectures content for a pre-trained keras model

#### System Requirements

I used my own gpu to train the models so you may want to try it out on a system with such capabilities or use your cpu but it
may take more time

  * python ">=3.11" [python.org](https://www.python.org/downloads/)
  * uv  ">=0.9.5"   [docs.astral.sh](https://docs.astral.sh/uv/getting-started/installation/)
  * docker  ">=29.1.3"  [docs.docker.com](https://docs.docker.com/engine/install/)
  * kubectl ">=v1.34.4" [kubernetes.io](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/)
  * kind    ">=v0.30.0"  [kind.sigs.k8s.io](https://kind.sigs.k8s.io/docs/user/quick-start/)
  * cuda-toolkit    ">=11.8" or ">=13.1, V13.1.80" [docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
    
    The reason why I included a minimum version (11.8) and a specific one (">=13.1, V13.1.80") is because I faced an error in my
    first notebook and the minimum version required is 11.8 or better, so I decided to use the specific version that matched my
    system drivers, just letting you know, either should work, if not, well you may have to find a solution on your own

    Note: I would suggest to not re-install your drivers to not affect your system as in the past for some reason I had problems
    including driver installation, in the end it is just a suggestion, choose what works best for you

You may want to check your RAM usage as audio samples are pre-processed in one go, when checking the notebooks you may want to
assign None or force a garbage collection to unused variables, I kept them to go back whenever required I am just letting you
know in advance so you can decide what to do, (RAM prices crazy bruh) searching for a course on optimization and fair resource
usage looks reasonable

#### Data

I got the data from [kaggle.com](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download) 

Go to Data Explorer and from Data folder expand the genres_original folder that is the data used, there you can find each genre and the audio samples in wav format

I wanted to experiment on my own with original data as you can see there are images and csv files available, so this time download genres_original folder which has the audio samples, to download you need a kaggle account and from download you only need the genres_original folder

How I did it: https://www.youtube.com/watch?v=pjShazdItm8

Once downloaded check if a folder named 'data' exists on the root of your uv environment if not create it, then move the genres_original folder with its contents to the data folder, once moved rename the genres_original folder to gtzan_ds. You have to end up with something like:

data/
....gtzan_ds/
........blues/
........classical/
........country/
...

This is to keep the expected folder structure to run the pre-processing of the audio samples

Later a folder named 'gtzan_np_arr' is going to be created inside of data folder, that new folder is used to store numpy arrays of the spectrograms, it stores the arrays once during spectrogram creation on their respective genre folder. Later you can skip the spectrogram creation and only load the numpy arrays (spectrograms) from the gtzan_np_arr folder

In case you want to try out other spectrogram parameters you can use a different folder just by changing the reference to save them just the parent folder in this case gtzan_np_arr, be aware of your resource usage and name references to not overwrite previous data and using the new value in the corresponding variable GTZAN_NP_ARR from the file [preprocess.py](preprocess/preprocess.py)

#### Models

During your first training mel spectrograms are computed for all audio samples, except 'jazz.00054.wav' which throws an error and fails to be processed, you can remove the file or keep it. After all spectrograms have been saved once you no longer need to run line 17 in [train.py](train_pipeline/train.py):

```python
    #pre_proc.save_spectrograms(GTZAN_DS, GENRES, GTZAN_NP_ARR)
```

Check the previous line is uncommented for your first run, It commented it because I was testing only loading previously saved data as I only saved spectrograms once in order to save time. Once done remember to comment back this line and only load saved data as I did

Unless you modify mel-spectrogram parameters in [preprocess.py](preprocess/preprocess.py) to try out a different set of values, this would generate a new set of spectrograms, in such case, I suggest you rename the previous gtzan_np_arr folder in the data folder to something else so that you keep previous spectrogram data or if you do not want to keep it simply delete the gtzan_np_arr folder just double check before deleting

Once you reach the stage of training a model check if a folder named 'models' exists on the root of your uv environment if not create it, is used to store the models after each checkpoint

By default the time window uses values in line 36 [model.py](modeling/model.py) from list:

```python
    TM_MASKS = [25, 75]
```

Be aware that number of computed models increases as more windows are added to this list, so it would take more time, it would be reasonable to use a single time window unless you have enough resources to include several values

To change the name of the referenced models you can edit from line 7 [train.py](train_pipeline/train.py) constants:

```python
    KERAS_MODEL_NAME = "keras_model_checkpoint.keras"
    ONNX_MODEL_NAME = "desired_onnx_model_name.onnx"
```

By default the values of each constant are set to the file names in the models folder which are a keras checkpoint (required to export) and onnx exported keras models (required for deployment)

The new onnx model name has to be reflected in the Dockerfile as well, by default it uses the already provided onnx file in models

#### Cloning

Create a folder to clone there

```bash
    mkdir cap_one_cleall
  
    cd cap_one_cleall
```

Clone project to cap_one_cleall/

```bash
    git clone https://github.com/cleall/capstone_one
```

Navigate to project folder capstone_one

```bash
    cd capstone_one
```

Install python version for project using uv

```bash
uv python install 3.11
```

Create venv using previously installed python version

```bash
    uv venv --python 3.11
```

Activate the environment

```bash
    source .venv/bin/activate
```

Install project dependencies

```bash
    uv sync --locked
```

Friendly reminder do not forget to add the gtzan data to the data folder as
mentioned in [Data](#Data) section

Video of how I did it: https://www.youtube.com/watch?v=3CwQoG5wZMg

#### Using train.py

Check:

  * All system dependencies have been satisfied

  * You have downloaded audio samples, and moved them to the data folder of your uv environment

  * You have renamed genres_original/ to gtzan_ds/

  * model folder exists in your uv environment

Once ready you can execute the following from the root of your uv environment:

```bash
    uv run -m train_pipeline.train.py
```

In case you get a ModuleNotFoundError: __path__ attribute not found on 'train_pipeline.train do:

```bash
    uv run -m train_pipeline.train
```

This is going to execute the training pipeline and process audio samples to create spectrograms of each music genre to use them later and train a model

You are going to see some messages on the screen indicating what step of the pipeline is being executed

By default it uses the already saved keras model and it exports it to onnx format

#### Test using uv

Once the model has been exported you can test it out using uv

Execute the following command from the uv environment root:

```bash
    uv run music_classifier.py
    .
    .
    .
    Uvicorn running on http://0.0.0.0:4444
```

Now to send a request execute the following command from the uv environment root:

```bash
    uv run music_classifier_test.py
    .
    .
    .
    Duration: 0.135 seconds

    Top prediction: metal (5.528)

    blues: -3.639
    classical: -11.141
    country: -4.232
    disco: -2.389
    hiphop: -5.136
    jazz: 4.435
    metal: 5.528
    pop: -0.238
    reggae: -0.707
    rock: 2.115
```

By default it sends a request using one of the available samples that were held on the test split

It is really interesting to compare the results as some music genres have strong similarities with others

You can change the request parameters by editing the [music_classifier_test.py](music_classifier_test.py):

```python
    request = {
        "genre": "metal",
        "name": "metal_testnpy.npy"
    }
```

To change the parameters take a look at the gtzan_test_data/ folder in data folder. This folder is created during training and it contains a sample of each music genre from the test split

Just change genre value for one of the available music genres folder names e.g. disco and name value to the name of the file inside of the selected music genre e.g. disco_testnpy.npy, you can notice that the only thing that changed was the music genre in both parameters

#### Dockerimage

After your first uv run and test it is time to create the docker image

First stop uvicorn, use CTRL+C to stop its execution from the terminal running its process

```bash
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:4444 (Press CTRL+C to quit)
INFO:     127.0.0.1:36198 - "POST /predict HTTP/1.1" 200 OK
^CINFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process
```

Now important to mention if you changed onnx model name make sure you update it in Dockerfile, same for any other resource referenced in Dockerfile

To build the Dockerimage execute the following command (use the tag of your preference):

```bash
    sudo docker build -t capstone_one_cleall:v1 .
```

To run Dockerimage in interactive mode using port 4444 from image and host execute the following command:

```bash
    sudo docker run -it --rm -p 4444:4444 --env PORT=4444 capstone_one_cleall:v1
```

#### Test Dockerimage

Execute the music_classifier_test.py with uv as follows:

```bash
    uv run music_classifier_test.py
    .
    .
    .
    Duration: 0.135 seconds

    Top prediction: metal (5.528)

    blues: -3.639
    classical: -11.141
    country: -4.232
    disco: -2.389
    hiphop: -5.136
    jazz: 4.435
    metal: 5.528
    pop: -0.238
    reggae: -0.707
    rock: 2.115
```

If the request remained with default values the result is the same as above if not well results may vary

#### Video interaction uv and dockerimage

Video interaction using uv, docker: https://www.youtube.com/watch?v=N_uPd-l75Dk

#### Kubernetes

To follow these steps it is required that docker, kubectl and kind are installed and working on your system

Create a kubernetes cluster

```bash
    sudo kind create cluster --name cl-capone-cleall
```

You can check cluster info

```bash
    sudo kubectl cluster-info
```

Be patient until cluster status reaches a Ready state

```bash
    sudo kubectl get nodes
```

This way you can identify the version used and avoid confusion with Dockerimages

Important: If you are using a different name and tag for your Dockerimage you have to reflect them in file [deployment.yaml](k8s/deployment.yaml)

spec:\
    containers:\
    - name: music-classifier\
    image: dockerimage_name:tag

In case you miss this the deployment is more likely to fail as it is not going to be able to load the correct Dockerimage

Now you have to load the Dockerimage to kind

```bash
    sudo kind load docker-image capstone_one_cleall:v1 --name cl-capone-cleall
```

It may take some minutes to load

###### Create Deployment manifest

Review file content here: [deployment.yaml](k8s/deployment.yaml)

I made a few changes from the one I created for the kubernetes workshop:

  * automountServiceAccountToken: false, Basically adhere to the principle of least privilege
  * image: capstone_one_cleall:v1, use of specific version is suggested to identify the image used easily, latest can cause confusion

All were suggested by SonarQube I just applied the fix I considered best

Note: The cpu, memory and delay values used are determined by how your system behaves they may require to be adjusted further.

Once ready apply it, from root of your uv environment do:

```bash
    sudo kubectl apply -f k8s/deployment.yaml
```

Wait for the message confirmation after creation and now check the deployment, status has to reach Running and Ready reflect 1/1 when using get pods, it may take a while

```bash
    sudo kubectl get deployments
    sudo kubectl get pods
    sudo kubectl describe deployment music-classifier
```

Again be patient as deployment.yaml configuration is applied and completed

Meanwhile you can review the logs

```bash
    sudo kubectl logs -l app=music-classifier --tail=30
```

The output is similar to the output seen when running uvicorn or the Dockerimage when deploying the service for testing.

###### Create Service manifest

Review file content here: [service.yaml](k8s/service.yaml)

The approach looks like the one applied in homework 10

Once ready apply it, from root of your environment do:

```bash
    sudo kubectl apply -f k8s/service.yaml
```

Wait for the message confirmation after creation and now check the service, you should see music-classifier with a LoadBalancer type pending for an external ip

```bash
    sudo kubectl get services
    sudo kubectl describe service music-classifier
```

#### Test service is working

Test the service forwarding port 4444 on the host to the port 4444 on the service, or the ones you used, if Dockerimage exposed port changed, you have to build the Dockerimage
again reflecting the change in [music_classifier.py](music_classifier.py) and [entrypoint.sh](entrypoint.sh) and use the following command consistently with your changes

```bash
    sudo kubectl port-forward service/music-classifier 4444:4444
```

You should see an output similar to the following

```bash
    Forwarding from 127.0.0.1:4444 -> 4444
    Forwarding from [::1]:4444 -> 4444
```

Now send a request from a different terminal:

```bash
    uv run music_classifier_test.py
```

The same result appears (if no parameter changed) after sending a request as showed during [Test using uv](#test-using-uv)

In case the connection closes after you send the request it may be required to assign more memory, cpu or both.

#### Use Autoscaling

To use autoscaling in homework 10 I had to get the metrics server and patch it, so I am going to do just that here, on a terminal do

```bash
    sudo kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

To use it without tls do the following, patch metrics server

```bash
    sudo kubectl patch -n kube-system deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

Confirm metrics server deployment is present

```bash
    sudo kubectl get deployment metrics-server -n kube-system
```

You must see READY 1/1, UP-TO-DATE 1, AVAILABLE 1

Check the status of the metrics server

```bash
    sudo kubectl get pods -n kube-system | grep metrics-server
```

You must see the id of the metrics server 1/1 Running

While it deploys review the Horizontal Pod Scaling configuration from file [hpa.yaml](k8s/hpa.yaml) you can adjust the parameters, I tried to keep them almost identical to the homework

Once the server has been installed and running you can try applying the horizontal pod scaling

To apply the hpa configuration do

```bash
    sudo kubectl apply -f k8s/hpa.yaml
```

Wait for the message confirmation after creation and get the hpa information use

```bash
    sudo kubectl get hpa
    sudo kubectl get pods -n kube-system
```

Be aware of the TARGET column it should display a numerical value

  * cpu: 0%/20%

In case it shows

  * cpu: unknown%/20%

There may be an issue with the metrics server where the hpa is unable to retrieve metrics from the metrics server. Either server was not properly installed or running. 

If the pods targeted by the hpa do not have resource requests and limits defined the hpa cannot calculate cpu or memory utilization.

Resources have to be Ready with Status Running, in particular your metrics server

#### Testing the Horizontal Pod Autoscaler

Stop the port forwarding of service/music-classifier, from the terminal executing it do CTRL+C

To test the hpa take a look at [load_test.py](/load_test.py) it seems reasonable to review the test parameters before you execute the test by default it sends 400 requests
and uses 4 workers, I left it this way as my cpu has 4 cores to work with and only 400 requests so it does not take long to complete. Feel free to adjust the load test so it triggers the horizontal pod scaling not exhausting your system resources.

Test the service forwarding port 4444 on the host to the port 4444 on the service, or the ones you used

```bash
    sudo kubectl port-forward service/music-classifier 4444:4444
```

Watch the behaviour of the hpa in other terminal

```bash
    sudo kubectl get hpa -w
```

You can also watch pod behavior in other terminal

```bash
    sudo kubectl get pods -w
```

Now go to the root of your uv environment an do:

```bash
    uv run load_test.py
```

In case requests fail to respond try lowering the number of requests and workers so that your system is not overwhelmed and all of them fail.

During test execution monitor hpa and pod behaviour from the previous commands to see how it scales.

You should see how hpa is applied when load reaches the specified cpu utilization this is seen by monitoring the output of the kubectl get hpa -w.

#### Video interaction kubernetes

I divided the interaction in two segments

In first part I just show that all the configuration has been applied to the kubernetes cluster and the metric server is up and running

  * Part one: https://www.youtube.com/watch?v=s11PMWGTc4g

In second part I show how the hpa behaved during the test

  * Part two: https://www.youtube.com/watch?v=VS0Xsad5Kb8

Thats all you reached the end of the README file.

Thanks for reviewing my project good luck!
