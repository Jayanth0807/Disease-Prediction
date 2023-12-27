The Basic idea of our project is to identify whether the patient is effected with disease.For that we had taken SARAS related diseases(COVID-19),dataset is prepared in such away that where we had taken 4 inputs they are,Temperature,Heartbeat,SpO2,ECG.Each inputs will have certain threshold values and abnormal values.

Then the respective sensors i.e,Heart-rate sensor which collects the real-time data of Heatbeat and SpO2,Temparature reading respectively,ECG sensor will collect the data through the patch cords.

So the data which was collected by the sensor are the analog values so we are using an I2C module to convert the values into discrete values which has to be given to the Micro-Controller Unit which is Raspberry-Pi 3 model b.
We developed a code in python platform and used Machine Learning algorithms,and KNN alogorithims and Fuzzy Logic.Here we used the KNN algorithm for nearest values that are related to our dataset.Fuzzy logic detemines who accurate the obtained value is.

after the successful running of our system if there are any Abnormalities then it will display "COVID SYMPTOMS IDENTIFIED",if there are no abnormalities then it shows as "PATIENT IS SAFE".By this project we can say that it can predict the disease based on the design with an accuracy upto 99%.

#Instructions

1.First connect the sensors with respect to the Raspberry-Pi,and a desired Raspberry-Pi display is used to display the output.

2.Then we Open our code which we dumped in our Raspberry-Pi MCU,there we have to select in the Menu the required file,where 	our code Exisits.

3.Open the code and RUN the code and then it will automatically starts to collect the sensors data,which are connected to the micro controller.

4.After successful training of the model the output will be displayed in the Raspberry Pi display,whether the patient is effected or not.
