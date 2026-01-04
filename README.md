This code is used as support for this article:
https://dzone.com/articles/training-a-neural-network-model-with-java-and-tens?preview=true

Using the Iris classification endpoint example(7.7,3.8,6.7,2.2 expected response: Iris-virginica):  
curl --verbose 'localhost:8095/tensorflow/iris-classify?sepalLength=7.7&sepalWidth=3.8&petalLength=6.7&petalWidth=2.2'

Using the object detection endpoint:  
curl --verbose --header 'Content-Type: application/octet-stream' --data-binary @src/main/resources/images/Beachgoers_enjoy_outing_at_the_beach.jpg 'localhost:8095/tensorflow/detect-objects'
