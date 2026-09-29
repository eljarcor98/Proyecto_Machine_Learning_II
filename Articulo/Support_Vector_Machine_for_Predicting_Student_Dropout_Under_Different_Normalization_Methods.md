2024 IEEE International Conference on Big Data (Big Data) 

# Support Vector Machine for Predicting Student Dropout Under Different Normalization Methods 

Gehan Boteju Leon Tang _Information Systems Department Information Systems Department University of Maryland Baltimore County University of Maryland Baltimore County_ Baltimore MD, United States Baltimore MD, United States gboteju1@umbc.edu leont1@umbc.edu 

Michael Scott Brown _Information Systems Department University of Maryland Baltimore County_ Baltimore MD, United States michaelb@umbc.edu 

**_Abstract_ — Student dropout in universities brings significant challenges that impacts both individual futures and institutional effectiveness. Early prediction of potential dropouts is crucial for timely intervention, but it is complex because of the nature of the problem influenced by diverse socioeconomic factors. This paper utlizies Support Vector Machines (SVMs) to predict student dropout with an emphasis on exploring the efficacy of various data normalization methods to optimize prediction accuracy. Using a dataset from the UC Irvine repository, this study compares 9 different normalization techniques such as Min Max Scaler, Standard Scaler, and Power Transformer, among others, to determine their impact on the predictive performance of SVMs. Results demonstrate substantial variations in model accuracy depending on the normalization method used to show the importance of detailed selection of data preprocessing techniques. The best normalization method was the One Hot Scaler which produced an average F1 score of 0.779. This work enhances the ability to identify at-risk students earlier but also the understanding of how data normalization influences predictive modeling in educational settings.** 

### **_Keywords—Support Vector Machine, Data Pre-processing, Data Normalization, Data Scaling, Predicting Student Success_** 

## I. INTRODUCTION 

Student success is a major objective of universities throughout the world. Universities strive to create adequate environments for students to thrive in as well as supply them with tools that help them succeed [1]. However, despite the investment in those resources, universities still see students drop out or graduate later than expected. A student’s success can be attributed to their ability to withstand rigorous coursework while balancing life. 

Universities are constantly exploring new ways to keep students on track and motivated. They have set up support structures like academic advisors and tutoring centers, along with mental health resources to help students navigate the pressures of college life. On top of that, the rise of digital tools such as learning management systems to a wealth of online resources has made it easier for students to find help when they need it [2].  Yet, even with these new features, each student's unique background and personal challenges influence their journey through university that impacts both their academic success and their path to graduation. 

## _A. Problem Statement_ 

Student dropout is a complex problem but if the school can detect signs early it is preventable. The difficulty in predicting student success is multifaceted. It is unclear what data attributes are needed to make predictions. Similarly, it is unclear what predictive models work best to predict student success. 

Many different factors influence a student’s decision to drop out. Various socioeconomic factors or just an overall lack of motivation to continue hinder a student’s ability to succeed [3]. The struggles of daily life as well as poor academic performance can compound and make college an uphill battle. 

## _B. Importance_ 

Making sure students graduate on time is very important. College is a large financial burden as well as a psychological burden [4]. Although having some college experience may benefit the student with some employers, generally not completing a degree is a waste of money. Too often this is financed debt, along with significant waste of time. If a student is not going to graduate they could have used the time attending school for other ventures like working or learning a trade. Failing out of college is a waste of time and money. 

## _C. Approach_ 

In this research, we used Support Vector Machines (SVM) to predict student dropout rates by analyzing a dataset and focused on different data normalization methods to optimize the prediction accuracy [5].  The normalization techniques explored include Scale, Max Absolute Scaler, One Hot Encoder, Quantile Transformer, Power Transformer, Ordinal Encoder, Label Encoder, and Min Max Scaler. Additionally, these methods adjust the data in many ways to improve the model's ability to interpret the features effectively. The dataset was compiled to address the issue of student dropouts and academic failures in higher education by identifying at-risk students at an early stage. Moreover, this method allows for timely intervention strategies to be used, based on insights into students' academic paths, demographics, and socioeconomic factors. The project categorizes student outcomes into three groups or classes: dropout, enrolled, and graduate, within the expected course duration with the goal to reduce dropout rates through targeted support. 

8633 979-8-3503-6248-0/24/$31.00 ©2024 IEEE Authorized licensed use limited to: University of Maryland Baltimore Cty. Downloaded on January 21,2025 at 17:17:21 UTC from IEEE Xplore.  Restrictions apply. 

## II. LITERATURE REVIEW 

## _A. Support Vector Machine_ 

The prediction of student dropout has become a critical research area because of the high impact it has on both individual students and educational institutions. Over the years, many machine learning methods have been explored to mitigate these conflicts with varying amounts of success. This section provides a review of relevant literature focusing on the application of SVMs in predicting student dropout with an emphasis on data normalization techniques. 

SVMs are a sophisticated type of supervised machine learning that is used to classify data into distinct categories. As shown in Figure 1, this hyperplane maximizes the margin between support vectors to ensure clear classification boundaries. These models work by finding the best hyperplane in an N-dimensional space (N being the number of features) that maximizes the margin between the data points of each class which are support vectors. These points are important as they find the hyperplane’s position and direction. To manage nonlinear data relationships, SVMs use the kernel trick, which allows the hyperplane to separate classes in a higher dimensional space without direct computation. 

Data normalization is important in SVM applications to ensure that each feature equally impacts the hyperplane’s assurance. Techniques such as scaling features to a range of 0-1 or standardizing them to zero mean and unit variance are used to prevent one feature from dominating. Additionally, in predicting student dropout, this normalization allows SVMs to analyze widespread data such as grades, attendance records, and socioeconomic background effectivity. This balanced method improves the model’s way to read from the training data to new and unseen data to predict accurately in educational settings. 



<!-- Start of picture text -->
Margin<br>+<br>C<br>‘ Loko {<br>stro ect 232<br>\ lOS°<br>° \<br>fone) $ °° a<br>ae) 50 Decision Boundary<br><!-- End of picture text -->

Fig. 1. Support Vector Machine 

## _B. Importance of Predicting Student Dropout_ 

The prediction of student dropout has become a focal point in educational data mining because of the rising conflicts over retention rate and academic success. Early signs of at-risk students allow institutions to use targeted interventions which are needed for decreasing dropout rates and increasing academic 

performance. Mduma et al. [6] present the importance of using machine learning models, such as SVMs to determine datasets that analyzes socio-economic factors, academic records, and behavioral patterns. The study presents the need for models that can uphold challenges by institutes especially in developing countries where resources are typically limited. 

## _C. Role of Support Vector Machines in Predictive Modeling_ 

SVMs are widely known because of their ability in handling classification tasks which make them a top choice for predicting student dropout. SVMs operate by finding the optimal hyperplane that separates different classes which is effective in binary classification problems like predicting if a student will drop out or continue their studies. Shahrir et al. [7] review many data mining techniques and confirm the efficacy of SVMs in educational data mining emphasizing the prediction of student performance. Moreover, Sharma et al. [8] gives an analysis of SVMs in predicting primary school dropouts showing that when combined with proper data preprocessing, SVMs can greatly improve prediction accuracy. 

In higher education, Hoffait and Schnys [9] used SVMs for the early detection of university students with potential academic difficulties. Their research showed that SVMs outperformed traditional statistical models in predicting student dropout especially when used in combination with large datasets that had academic and sociodemographic variables. This connects with the findings of Beaulac and Rosenthal [10] who applied SVMs and other machine learning techniques to predict university students’ academic success and major selection. 

## _D. Impact of Data Normalization on Support Vector Machines Performance_ 

Data Normalization is important to the success of SVMs based predictive models as it ensures that all features contribute to the model’s decision boundary. Normalization techniques such as Min Max Scaler, Standard Scaler, and Power Transformer have been shown to improve the accuracy of SVM models. Roy and Farid [11] conducted a comparative study of these normalization methods and found that they improve the predictive performance of SVMs and the stability of models across different datasets. 

In their review, Rastrollo-Guerrero et al. [12] emphasize the importance of data preprocessing in machine learning for predicting student performance. They argue that the choice of normalization technique can have a big impact on the model’s ability to generalize across different educational contexts. Additionally, this is consistent with the findings of Maheshwari et al. [13] , who looked into the impact of data normalization in their comparative analysis of machine learning models and found that SVMs with effective normalization techniques consistently outperformed other models in predicting dropouts. 

## _E. Comparative Analysis of Machine Learning_ 

While SVMs are very effective, it is crucial to compare their performance with other machine learning models to allow the best predictive accuracy. Masheshwari et al. [13] compared SVMs with neural networks, random forest, and logistic regression and found that SVMs provided the highest accuracy in predicting student dropout. Furthermore, this was backed up by Sharma et al. [8], which found that SVMs working with 

8634 Authorized licensed use limited to: University of Maryland Baltimore Cty. Downloaded on January 21,2025 at 17:17:21 UTC from IEEE Xplore.  Restrictions apply. 

proper data preprocessing outperformed other models in primary school dropout prediction. Similarly, Hoffait and Schyns [9] demonstrated that SVMs were effective in predicting dropout when compared to traditional models using datasets with imbalanced class distributions. 

Thammasiri et al. [14]  also addressed the imbalanced class distribution problem in dropout prediction which shows the challenges faced by models like SVMs when dealing with such datasets. Their study imposes that using SVMs with techniques to address class imbalance such as Synthetic Minority Oversampling Technique (SMOTE) can further improve model performance. This approach is supported by Chawla et al. [15] who introduced SMOTE to improve predictive accuracy of models like SVMs in cases of imbalanced data. 

## III. METHODOLOGY 

## _A. Research Method Employed_ 

Our approach to predicting student dropout involves creating a SVM models that apply normalization methods. Transforming the dataset using a scaled method helps the SVM process the features [16]. The data is split into training and test data and the SVM is trained to create a predictive model. A common metric used in with this dataset is the average F1 scores of the 3 classes: graduated, dropped out and still enrolled. 

## _B. Specific Procedure Employed_ 

Because results could differ based upon what records are used for training and test data, each set of parameters were executed 50 times. Each execution generates an average F1 score calculated for the three classes. Executing each set of parameters 50 times produces an average of the average F1 scores. 

A SVM was trained on the training data. The domain values were normalized using one of the normalization methods shown in Table 1. So, each normalization method was used 50 times with different combinations of records for training and testing. When the SVM is used to predict student success on the test data the average F1 score is computed. Upon completion of the algorithm there is an average of the average F1 scores for each normalization method. 

The configuration parameters used for the SVM are as follows. The kernel was polynomial with degree of 3. All other parameters were scikit-learn default values. 

A T-Test is performed comparing the average of the average F1 scores to the SVM’s results without any normalization. Results with a p-value less than 0.05 show that the results with normalization are statistically different from the results without normalization. 

## _C. Format for Presenting Results_ 

The results will be shown in a table. Each normalization method will be represented by a row. The average of the average F1 scores for the 50 trials will be shown corresponding to the normalization method. A third column will contain the p value of the T-Test that compared results for that normalization method to results without normalization. 

The dataset contains 4424 records, each for a unique student. There are 36 domain attributes. They consist of student information like gender, age, mother’s occupation, unemployment rate and many more. The 37th column is the enrollment status: Graduated, Dropped out, Enrolled. 

TABLE I. TYPES OF DATA NORMALIZATION 

|**_Normalization_**<br>**_Method_**|**_Function_**|**_Formula (if used)_**|
|---|---|---|
|Standard Scaler|Normalizes data to<br>have a mean of 0<br>and a standard<br>deviation of 1.|𝑧 =<br>(𝑥 − 𝑢)<br>𝑠<br>𝑥 = 𝑠𝑎𝑚𝑝𝑙𝑒<br>𝑢 = 𝜇 𝑡𝑟𝑎𝑖𝑛𝑖𝑛𝑔 𝑑𝑎𝑡𝑎<br>𝑠=𝜎 𝑡𝑟𝑎𝑖𝑛𝑖𝑛𝑔 𝑑𝑎𝑡𝑎|
|Max<br>Absolute<br>Scaler|Takes the features<br>within a dataset and<br>scales them in a<br>range of-1 to 1.||
|One Hot Encoder|Converts<br>categorical<br>variable(s) into a<br>binary vector<br>representation<br>where a unique<br>binary vector<br>represents each<br>category.||
|Scale|<br>Features within a<br>dataset are<br>standardized among<br>anaxis.||
|Quantile<br>Transformer|Transforms the<br>features to follow a<br>uniform or a normal<br>distribution by<br>ranking the data and<br>then mapping them<br>to the desired<br>distribution using<br>the quantile<br>function.||
|Ordinal Encoder|Converts<br>categorical features<br>into integer codes<br>based on the order<br>(or rank) of the<br>categories, which is<br>manually specified<br>or determined from<br>the data.||
|Min Max Scaler|<br>Adjusts data values<br>to fit within a set<br>range, usually 0 to<br>1.|𝑥 𝑠𝑡𝑑=<br>(X −X. min(axis = 0))<br>(𝑋. max(𝑎𝑥𝑖𝑠= 0) −<br>𝑋. min (𝑎𝑥𝑖𝑠= 0))<br>⬚<br>𝑋 𝑠𝑐𝑎𝑙𝑒𝑑= 𝑋 𝑠𝑡𝑑∗<br>(𝑚𝑎𝑥−𝑚𝑖𝑛) +<br>𝑖|
|Label Encoder|Convert categorical<br>labels into a<br>numeric format,<br>assigning a unique<br>integer to each class<br>of thelabel.|𝑚𝑛|
|Power<br>Transformer|Applies a power<br>transformation to<br>each feature to<br>make the data more<br>Gaussian-like.||



8635 Authorized licensed use limited to: University of Maryland Baltimore Cty. Downloaded on January 21,2025 at 17:17:21 UTC from IEEE Xplore.  Restrictions apply. 

## _D. Resources Required_ 

Conducting this research required the following resources. All coding was completed in Python with the scikit-learn libraries, which leverages numpy. The dataset was obtained from the University of California Irvine Dataset Repository [17]. The code was executed on a Personal Computer. 

## IV. RESULTS 

The final output was a .csv file that saved the F1 score from each SVM trial for each normalization method employed as well as an SVM model with no data normalization method. The final column in the csv file contains the average F1 score from each data normalization method. Table 2 is a summarized table of the output that contains only the F1 scores and the data normalization method. 

|TABLE II.|AVERAGEF1 SCOREAFTER|50 TRIALS|
|---|---|---|
|**_Normalization_**<br>**_Method_**|**_Average F1 Score_**|**_P value_**|
|Standard Scaler|0.714825|>0.00000|
|Max Absolute Scaler|0.764814|>0.00000|
|One Hot Encoder|0.778554|>0.00000|
|Scale|0.713876|>0.00000|
|Quantile Transformer|0.754102|>0.00000|
|Ordinal Encoder|0.511751|0.00027|
|Min Max Scaler|0.764068|>0.00000|
|Label Encoder|0.499616|0.905|
|Power Transformer|0.724836|>0.00000|
|No<br>Normalization<br>Method|0.5|>0.00000|



The data normalization methods that proved to be exceptional include Min Max Scalar with a F1 score of .76, Max Absolute Scalar with a F1 score of .76, One Hot Encoder with a F1 score of .77, and Quantile Transformer with a F1 score of .75. Data normalization methods that produced good results were Standard Scalar with a F1 score of .71, scale with a F1 score of .71, and Power Transformer with a .72. Data normalization methods with bad F1 scores include Ordinal Encoder with a F1 score of .51 and Label Encoder with a F1 score of .49. 

When comparing the F1 scores of exceptional methods to the F1 scores of no normalization methods there is a clear difference between the two. There is an average of a .27 difference between exceptional data normalization methods and no data normalization method. 

## V. CONCLUSION 

This study aimed to accurately predict student success by applying various data normalization methods to the dataset before creating SVM. The goal was to determine if the various data normalization methods would improve the F1 score of the SVM opposed to creating SVM without data normalization methods. 

Results showed that 7 of the 9 data normalization methods used did improve the accuracy of the F1 score. The remaining 2 data normalization methods gave outputs that were closer to a SVM with no normalization method. Of the data normalization methods employed, One Hot Encoder was the best as it produced an average F1 score of 0.778554 or .78 rounded up. 

To further expand upon this research it would be beneficial to apply other normalization methods. In this study only 9 were used while other normalization methods exist. 

## REFERENCES 

- [1] V. Tinto, _Completing College: Rethinking Institutional Action_ , Chicago, IL: University of Chicago Press, 2012. 

- [2] E. Dahlstrom, J. D. Walker, and C. Dziuban, _ECAR Study of Undergraduate Students and Information Technology_ , EDUCAUSE Center for Analysis and Research, 2013. 

- [3] J. P. Bean, "Dropouts and Turnover: The Synthesis and Test of a Causal Model of Student Attrition," _Research in Higher Education_ , vol. 12, no. 2, pp. 155-187, 1980. 

- [4] S. L. DesJardins and B. P. McCall, "The Impact of the Gates Millennium Scholars Program on College Students’ Outcomes: A Regression Discontinuity Approach," _Economics of Education Review_ , vol. 38, pp. 124-135, 2014. 

- [5] S. Hussain, W. Zhu, W. Zhang, S. M. R. Abidi, and S. Ali, "Using Machine Learning to Predict Student Difficulties from Learning Management System Data," _Educational Technology & Society_ , vol. 21, no. 4, pp. 90-103, 2018. 

- [6] N. Mduma, K. Kalegele, and D. Machuve, "A survey of machine learning approaches and techniques for student dropout prediction," _Data Sci. J._ , vol. 18, pp. 1–10, 2019.https://doi.org/10.5334/dsj-2019-014. 

- [7] A. M. Shahiri, W. Husain, and N. A. Rashid, "A review on predicting student’s performance using data mining techniques," _Procedia Comput. Sci._ , vol. 72, pp. 414–422, 2015. 

- [8] M. P. Sharma, P. Naglia, and K. P. Sharma, "Performance analysis for predicting primary school dropouts: Identifying using machine learning optimal algorithm/method," _Int. J. Innov. Stud._ , 2024. 

- [9] A. S. Hoffait and M. Schyns, "Early detection of university students with potential difficulties," _Decis. Support Syst._ , vol. 101, pp. 1–11, 2017. [Online]. 

- [10] C. Beaulac and J. S. Rosenthal, "Predicting university students’ academic success and major using random forests," _Res. High. Educ._ , vol. 60, no. 7, pp. 1048–1064, 2019. [Online]. 

- [11] K. Roy and D. M. Farid, "An adaptive feature selection algorithm for student performance prediction," _IEEE Access_ , 2024. [Online]. 

- [12] J. L. Rastrollo-Guerrero, J. A. Gómez-Pulido, and A. Durán-Domínguez, "Analyzing and predicting students’ performance by means of machine learning: a review," _Appl. Sci._ , vol. 10, pp. 1042–1058, 2020. 

- [13] A. Maheshwari, A. Malhotra, and B. S. Hada, "Comparative analysis of machine learning models in predicting academic outcomes: insights and implications for educational data analytics," _IEEE Access_ , 2024. 

- [14] D. Thammasiri, D. Delen, P. Meesad, and N. Kasap, "A critical assessment of imbalanced class distribution problem: the case of predicting freshmen student attrition," _Expert Syst. Appl._ , vol. 41, no. 2, pp. 321–330, 2014. 

- [15] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: synthetic minority over-sampling technique," _J. Artif. Intell. Res._ , vol. 16, pp. 321–357, 2002. 

- [16] "Preprocessing Data," Scikit-learn: Machine Learning in Python, Available: 

- [17] V. Realinho, M. Vieira Martins, J. Machado, and L. Baptista, "Predict Students' Dropout and Academic Success," UCI Machine Learning Repository, 2021. https://doi.org/10.24432/C5MC89 

8636 Authorized licensed use limited to: University of Maryland Baltimore Cty. Downloaded on January 21,2025 at 17:17:21 UTC from IEEE Xplore.  Restrictions apply. 

