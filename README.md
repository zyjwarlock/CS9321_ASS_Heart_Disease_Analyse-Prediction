# COMP9321-Assignment 3

Project structure:

![M](https://ws3.sinaimg.cn/large/006tNc79ly1g224rfckx4j319w0lwmzp.jpg)

## Implement

1. Here frontend and backend are implemented together using python package: **DASH** by Plotly,  which is built on top of Plotly.js, React, and Flask. Following the Minimalism design, we could focus on visual effects of graphs and maching learning model. 

   The main differences between DASH and tradidional HTML+CSS are:

   1. 'html' is considered as a class, traditional html labels are functions of class 'html'. Hence frontend can be written in python file as

      ```python
      html.Div(
      			children=[html.Div([]),html.Div([])],
      			style={css style}
      )
      ```

   2. HTTP request and response implementation is similiar to Flask. Using python decorator wrapper to get users' inputs and then response.

2. For graphing part, we used **Plotly**. On one hand it could draw *interactive* plots easily, instead of  just statical picture(matplotlib). Another reason is that the Plotly code is compatible with DASH code. Which is much more efficient for this project.

3. Loose coupling is also a key factor in this project. Hence machine learning model here is wrapped as a class to keep structures  isolated. **Firstly it will detect whether model mymodel.pkl exists, if not, it will generate a new model.**

   1. Using Random Forest Algorithm to avoid overfitting and also performances well on large feature datasets. Also, when initilizating model instance, users can choose boosting or svm algorithms. All of them are initilized with optimal parameters(GridSearch) .

   2. For bonus part, Firstly choose the 8 top ranked  features. And then use KNN algorithm, which could achieve 1% - 2% accuracy rate, test by 10 folds cross validation. This is shown as a line chart. 

      1. Y axis: Accuracy rate
      2. X axis: Number of neighbours

      

## User Guide

Run [run.py](/var/folders/3x/n1g4t2ln0x70lgxbrcpk9r2c0000gn/T/abnerworks.Typora/625F838D-7B65-4815-A4AD-8F0C3E56E12C/run.py) , and type <http://127.0.0.1:8050> in the browser. 

There are four graphs in a single page:

1.  The first graph shows statistics for basic attributes by groups of gender and age. Users could do drag, zoom in/out, save ….in this interactive graph.
2. Graph 2 shows the top 5 attributes which are the key factor involving in heart disease.
3. A real time prediction, once user's input is detected, the graph will update and compare your figures with average level(the same gender with you).
4. Improvement part. By implementing dimension reduction algorithm, the accuracy could increase a little bit depends on the number of KNN neighbours.



## Getting Involved

- [Xuanxin Fang/z5142897](http://fxx.me)
- Ting Hu/z5144926
- Taiyan Zhu/z5089986
- Xingchen Guo/z5109353


