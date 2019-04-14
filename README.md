# cs9321

Project structure:

![M](https://ws3.sinaimg.cn/large/006tNc79ly1g224rfckx4j319w0lwmzp.jpg)

## Implement

1. Here frontend and backend are implemented together using python package: **DASH** by Plotly,  which is built on top of Plotly.js, React, and Flask. Following the Minimalism design, we could focus on visual effects of graphs and maching learning model. 

   The main differences between DASH and tradidional HTML+CSS are:

   1. 'html' is considered as a class, traditional html labels are functions of class 'html'. Hence frontend can be written in python file as

      ```python
      html.Div(
      			children=[
      				html.Div([]),
      				html.Div([])
      			],
      			style={css style}
      ```

   2. HTTP request and response implementation is similiar to Flask. Using python decorator wrapper to get users' inputs and then response.

2. For graphing part, we used **Plotly**. On one hand it could draw *interactive* plots easily, instead of  just statical picture(matplotlib). Another reason is that the Plotly code is compatible with DASH code. Which is much more efficient for this project.

3. Loose coupling is also a key factor in this project. Hence machine learning model here is wrapped as a class to keep structures  isolated. 

   1. what algorithm



## Performance






