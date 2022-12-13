Design a captcha cracker using ML. 

The model should initially handle single characters (numbers 0-9 and letters A-F) in 24x32 8-bit grayscale images. 
Your training set will include 80000 such PNG files distributed across the “training_set/{character}/” directories with the sub-directory name indicating the class of the images. 

In the "test_sets" directory yout will find three testing datasets. 
They are meant for evaluating your model and you should not train your models on them.
Study your model’s performance and behavior on the training + test sets and document your observations.

Find a way to employ your model to crack 4-character captchas on 64x32 8-bit grayscale images. 
You will find a dataset to test this functionality in “full_captchas” where each file will be named “{imageid}_{captchatext}.png”. 
Feel free to propose multiple approaches to this problem and describe their strengths and drawbacks.  