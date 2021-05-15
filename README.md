
![image](https://user-images.githubusercontent.com/51465352/118379394-6519e700-b5b0-11eb-82d7-cba2c813c031.png)

# Face_Recognition

In this project I used the libraries Face Recognition and openCV. Follow the instructions to run in your computer.

# How to install
First of all, clone this repository in your favorite directory, using:

    git clone https://github.com/Matheus-Barros/Face_Recognition.git

I recomend you to to install anaconda, to be able to install all the libraries that I used in this project. With that said, with your anaconda prompt opened, got tho the directory where the file Face_Rec-ENV.yml it is. Now write in the anaconda prompt:

    conda env create -f Face_Rec-ENV.yml

Wait till the installation ends. After that, you must activate this new enviroment. To do that, write in the anaconda prompt:

    conda activate Face-Rec

Done! Here we have all that we need to make this script run! Now in the anaconda prompt, go to the scripts folder, and then run the Face_Recognition.py.

# Configurations

You can add new people to recognize. To do that, just create a new folder in the 'Faces' folder, with the name of the person. Then add at least 5 pictures of the person face. You also need to write the name of the person, in the list 'people'in the file Face_Recognition.py line 28. There you can choose who you wanna recognize. The name must be the same of the new folder created.
