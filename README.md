# Evaluation-of-ML-Workloads-on-Different-Processing-Units
This work is done as a Capstone project for the course EECE 5643: Simulation and Performance Evaluation at Northeastern University, Boston. The team that worked on this project is Raksha Ramkumar, Balaji Sundareshan and Akanksha Agnihotri. This project aims at understanding the impact of various hardware such as CPU, GPU and multi-GPU on different types of Machine Learning and Deep Learning models in terms of timing and accuracy. 

## Motivation
In this project, we analyze the performance and execution time of certain machine learning and 
deep learning architecture on the different hardware: CPU, GPU, and multi-GPU to understand 
the efficiency that each of these hardware brings to the training and inference period. This 
analysis is very crucial as it will help the researchers and practitioners to make well-informed 
decisions about which hardware suits a particular type of algorithm.

We also implement and analyse the multi-GPU setup for training various models. This is important because it allows us to examine the communication cost between the 
GPUs and how it affects the overall training timing of our algorithms.

## Experimental Setup
The following table contains details of the different hardware that we will be using to compare 
the execution time of the different algorithms:

![image](https://github.com/RakshaRamkumar/Evaluation-of-ML-Workloads-on-Different-Processing-Units/assets/63054940/9e89228e-0d78-4c1d-bb9c-c51cfb842ae6)

The entire project was programmed in Python using the PyTorch framework.

## Datasets and Workloads
3 Machine Learning Algorithms namely Linear Regression, Logistic Regression and Decision Trees and one Deep Learning based approach, namely Resnet-18 were used as workloads. For the machine learning approaches, the datasets used were UCI Higgs Dataset for classification and UCI Year Prediction MSD for regression. CIFAR 100 Dataset was used to train the Resnet-18 architecture.

## Results and Analysis
The two major metrics that we are considering are the training time and the accuracy of the different 
algorithms across the 3 hardware setups. We are considering the training time as the average training 
time per epoch as we have algorithms of varying complexities, and it would not be a fair comparison to 
measure the total training time since certain algorithms might take longer to learn the task at hand. We 
observe that even in such cases, the complexity of the algorithm will be balanced out with the timing 
improvements due to the hardware (such as GPU/multi-GPU).

The other parameter that we are interested in is the accuracy of the algorithms. The accuracy of the 
trained models across the 3 set-ups should be comparable. If a particular hardware is offering speed-up 
in terms of the training time but at the cost of accuracy, then depending on the application and context, 
one could argue that hardware acceleration could be traded for accuracy. So, tracking accuracy across 
CPU, GPU and multi-GPU is a key evaluation metric in this project.

### Linear Regression
We used the multi-layer perceptron to implement Linear Regression. Here there were 4 Linear 
layers each having 180 nodes followed by 2 Linear layers having 90 nodes and activation function of Relu 
was applied to each layer of Linear layer till the time we get one input and only one output node. The evaluation output for this algorithm is represented in the following graph:

![image](https://github.com/RakshaRamkumar/Evaluation-of-ML-Workloads-on-Different-Processing-Units/assets/63054940/d6ed90b5-580a-4de6-9945-a7f789a0ec6d)

### Logistic Regression
Similar architecture was used to implement the Logistic Regression, except the last layer was a Sigmoid layer since this is a classification task. The evaluation output for this algorithm is represented in the following graph:

![image](https://github.com/RakshaRamkumar/Evaluation-of-ML-Workloads-on-Different-Processing-Units/assets/63054940/7f927421-00fc-4126-8e16-74aa3d4b00d0)


### Decision Trees
Here, the decision tree is being used for the classification task on the UCI Higgs dataset. For the decision 
tree implementation, we have used NVIDIA’s XGBoost as it provides support for tree-based algorithms. 
Task is used for parallelization/multi-GPU implementation. It is a parallel computing library built on 
Python that manages the distributed workers and excels at handling large, distributed data science 
workflows. The objective function used is Binary Logit Raw and the tree method is Histogram. 
The results of the training time and testing accuracy across the different hardware is as follows:

![image](https://github.com/RakshaRamkumar/Evaluation-of-ML-Workloads-on-Different-Processing-Units/assets/63054940/3d96cac6-f27e-4126-9ef6-9d0992823d2d)

### Image classification using Resnet-18
For the Resnet-18 network trained for the image classification task, the trainable parameters are saved in the datatype 
float32. As this datatype contains high precision and takes more memory, we experimented 
with using float16 in some layers, particularly for the image classification task. Having layers of 
different datatype in an architecture is called Mixed Precision. The package used to implement 
Mixed Precision in pytorch is Apex.

As FP16 datatype has less number range compared to FP32 datatype, the memory usage of FP16 is lower 
comparatively. Due to this memory reduction, we can increase the batch size while training and 
inference. As precision is less, the time taken by FP16 model is less compared to FP32. The following graph represents the improvement in training time across the different precisions:

![image](https://github.com/RakshaRamkumar/Evaluation-of-ML-Workloads-on-Different-Processing-Units/assets/63054940/d9264fab-0c80-4569-ba78-ca5018d6c747) 

## Conclusion
![image](https://github.com/RakshaRamkumar/Evaluation-of-ML-Workloads-on-Different-Processing-Units/assets/63054940/bcccbf30-5fce-4026-8866-91a415dcf35f) 

The above table shows the performance improvements of all the workloads on 1 GPU, 2 GPUs and 4 
GPUs compared to CPU.

-  In the case of linear regression, as the workload is small, the speedup in single GPU compared 
to CPU is less. But with the increase in the number of GPUs, there is a significant improvement. 
Similar behavior is seen in the logistic regression workload. These two workloads are 
implemented in PyTorch framework.

-  So, if the workload is low-compute intensive and the dataset is large, then multiple GPUs are 
the desired configuration. If the workload is low-compute intensive and the dataset is small, it is 
better to use just a CPU as data parallelism might not be required.

- For Decision Trees, the performance is similar across multiple GPUs and performance 
significantly better than CPU. So, for this workload single GPU is the ideal configuration.

-  For CNN workload, as the deep learning training is compute intensive, using GPUs is highly 
recommended. The number of GPUs is dependent on the dataset size with more GPUs 
preferred for larger dataset.

-  If the memory of individual GPU is less or if the importance on the training/inference speed of 
the model overshadows the accuracy of the model, then mixed-precision configuration is more 
desired.
