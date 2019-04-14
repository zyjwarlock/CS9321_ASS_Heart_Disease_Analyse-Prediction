import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import plotly.figure_factory as ff

import numpy as np
import pandas as pd

import model as mm

M = mm.MyModel()

df = pd.read_csv('processed.cleveland.data')
df.columns = [
	'age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol',
	'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
	'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels',
	'thalassemia', 'target'
]

df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)
df = df.applymap(lambda x: 0 if x == '?' else x)

fig_2 = {
	'data': [
		{
			'x': [1, 2, 3],
			'y': [4, 1, 2],
			'type': 'bar',
			'name': 'SF'
		},
		{
			'x': [1, 2, 3],
			'y': [2, 4, 5],
			'type': 'bar',
			'name': u'Montréal'
		},
	],
	'layout': {
		'height': 300,
		'width': 200,
		'margin': {
			'l': 10,
			'b': 20,
			't': 0,
			'r': 0
		}
	}
}


def q1():
	D = {
		(1, 1): 'chest_pain_type',
		(1, 2): 'resting_blood_pressure',
		(1, 3): 'cholesterol',
		(1, 4): 'fasting_blood_sugar',
		(2, 1): 'rest_ecg',
		(2, 2): 'max_heart_rate_achieved',
		(2, 3): 'exercise_induced_angina',
		(3, 1): 'st_depression',
		(3, 2): 'st_slope',
		(3, 3): 'num_major_vessels',
		(4, 1): 'thalassemia'
	}



	q1_fig = plotly.tools.make_subplots(
		rows=4,
		cols=4,
		shared_xaxes=True,
		subplot_titles=[
			'CPT', 'RBP', D[(1, 3)], 'FBS', 'RE', 'MHRA', 'EIA', '', D[(3, 1)],
			D[(3, 1)], 'NMV', '', D[(4, 1)], '', '', ''
		]
	)

	q1_df = df.astype('float64')
	q1_df['sex'] = q1_df['sex'
						].apply(lambda x: 'Male' if x == 1.0 else 'Female')

	# for k in q1_df.keys():
	for k in D.keys():
		trace = ff.create_facet_grid(
			q1_df,
			x='age',
			y=D[k],
			color_name='sex',
			#         show_boxes=False,
			marker={'size': 5,
					'opacity': 1.0},
			colormap={
				'Male': 'rgb(165, 242, 242)',
				'Female': 'rgb(253, 174, 216)'
			},
			facet_col_labels='name',
			facet_row_labels='name',
			#         ggplot2=True
		)
		#     print(k[0], k[1])
		for f in trace.data:
			q1_fig.append_trace(f, k[0], k[1])


	q1_fig['layout'].update(
		height=550,
		width=550,
		showlegend=False,
		title='<b>Statistics of Attributes<b>',
#		title="<b>Feature importance</b>",
		titlefont=dict(size=20, color='rgb(23,203,203)'),
	)
	return q1_fig


def q2():
	importance = M.feature_importance()
	values = [round(e[-1], 4) for e in importance]
	phases = [e[0] for e in importance]

	colors = [
		f'rgb({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)})'
		for i in range(5)
	]
	n_phase = len(phases)
	plot_width = 300
	section_h = 100
	section_d = 10
	unit_width = plot_width / max(values)
	phase_w = [int(value * unit_width) for value in values]
	height = section_h * n_phase + section_d * (n_phase - 1)
	shapes = []
	label_y = []

	for i in range(n_phase):
		if (i == n_phase - 1):
			points = [
				phase_w[i] / 2, height, phase_w[i] / 2, height - section_h
			]
		else:
			points = [
				phase_w[i] / 2, height, phase_w[i + 1] / 2, height - section_h
			]

		path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

		shape = {
			'type': 'path',
			'path': path,
			'fillcolor': colors[i],
			'line': {
				'width': 1,
				'color': colors[i]
			}
		}
		shapes.append(shape)
		label_y.append(height - (section_h / 2))
		height = height - (section_h + section_d)

	label_trace = go.Scatter(
		x=[-350] * n_phase,
		y=label_y,
		mode='text',
		text=phases,
		textfont=dict(color='rgb(200,200,200)', size=10)
	)

	value_trace = go.Scatter(
		x=[350] * n_phase,
		y=label_y,
		mode='text',
		text=values,
		textfont=dict(color='rgb(200,200,200)', size=10)
	)

	data = [label_trace, value_trace]

	layout = go.Layout(
		title="<b>Feature importance</b>",
		titlefont=dict(size=20, color='rgb(23,203,203)'),
		shapes=shapes,
		height=560,
		#		width=800,
		width=500,
		showlegend=False,
		paper_bgcolor='#FFFFFF',
		plot_bgcolor='#FFFFFF',
		xaxis=dict(
			showticklabels=False,
			zeroline=False,
		),
		yaxis=dict(showticklabels=False, zeroline=False)
	)

	fig = go.Figure(data=data, layout=layout)
	return fig


q1_fig = q1()
q2_fig = q2()

app = dash.Dash()
app.layout = html.Div(
	children=[
		# html.Link(href='/assets/stylesheet.css', rel='stylesheet'),
		html.H1(children='Sichuan Hoptop', style={
			'text-align': 'center'
		}),
		html.H1(),
		html.H1(),
		html.Div(
			children=[
				html.Div([dcc.Graph(id='TL', figure=q1_fig)]),
				html.Div([dcc.Graph(id='TR', figure=q2_fig)])
			],
			style={
				'columnCount': 2,
				'width': '1000px',
				'margin-right': 'auto',
				'margin-left': 'auto',
			}
		),
		html.H1(),
		html.H1(),
		html.H2(
			children='Heart disease prediction', style={
				'text-align': 'center'
			}
		),
		html.H5(
			children='Leave the field blank if not sure',
			style={
				'text-align': 'center'
			}
		),
		html.Div(
			[
				html.P(
					[
						html.Label('1. age'),
						dcc.Input(id='age', value=50, type='number'),
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				),
				html.P(
					[
						html.Label('2. sex'),
						dcc.Dropdown(
							id='sex',
							options=[
								{
									'label': 'Male',
									'value': 1
								}, {
									'label': 'Female',
									'value': 0
								}
							],
							value=1
						)
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				),
				html.P(
					[
						html.Label('3. chest pain type'),
						dcc.Dropdown(
							id='chest_pain_type',
							options=[
								{
									'label': 'typical angin',
									'value': 1
								},
								{
									'label': 'atypical angina',
									'value': 2
								},
								{
									'label': 'non-anginal pain',
									'value': 3
								},
								{
									'label': 'asymptomatic',
									'value': 4
								},
							],
							value=3.17
						)
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				),
				html.P(
					[
						html.Label('4. resting blood pressure'),
						dcc.Input(
							id='resting_blood_pressure',
							value=131,
							type='number'
						),
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				),
				html.P(
					[
						html.Label('5. serum cholestoral in mg/dl'),
						dcc.Input(id='cholesterol', value=239, type='number'),
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				),
				html.P(
					[
						html.Label('6. fasting blood sugar > 120 mg/dl'),
						dcc.Dropdown(
							id='fasting_blood_sugar',
							options=[
								{
									'label': 'Yes',
									'value': 1
								}, {
									'label': 'No',
									'value': 0
								}
							],
							value=0
						)
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				),
				html.P(
					[
						html.Label('7. resting electrocardiographic'),
						dcc.Dropdown(
							id='rest_ecg',
							options=[
								{
									'label': 'normal',
									'value': 0
								},
								{
									'label': 'having ST-T wave abnormality',
									'value': 1
								},
								{
									'label': 'showing probable',
									'value': 2
								},
							],
							value=1
						)
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				),
				html.P(
					[
						html.Label('8. maximum heart rate achieved'),
						dcc.Input(
							id='max_heart_rate_achieved',
							value=148,
							type='number'
						)
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				),
				html.P(
					[
						html.Label('9. exercise induced angina'),
						dcc.Dropdown(
							id='exercise_induced_angina',
							options=[
								{
									'label': 'Yes',
									'value': 1
								}, {
									'label': 'No',
									'value': 0
								}
							],
							value=0.3
						)
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				),
				html.P(
					[
						html.Label('10. ST depression'),
						dcc.Input(id='st_depression', value=1, type='number')
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				),
				html.P(
					[
						html.Label('11. ST slope'),
						dcc.Dropdown(
							id='st_slope',
							options=[
								{
									'label': i,
									'value': i
								} for i in range(4)
							],
							value=1.6
						)
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				),
				html.P(
					[
						html.Label('12. number of major vessels'),
						dcc.Dropdown(
							id='num_major_vessels',
							options=[
								{
									'label': i,
									'value': i
								} for i in range(4)
							],
							value=0
						)
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				),
				html.P(
					[
						html.Label('13. thalassemia'),
						dcc.Dropdown(
							id='thalassemia',
							options=[
								{
									'label': 'normal',
									'value': 3
								}, {
									'label': 'fixed defect',
									'value': 6
								}, {
									'label': 'reversable defect',
									'value': 7
								}
							],
							value=5
						)
					],
					style={
						'width': '250px',
						'margin-right': 'auto',
						'margin-left': 'auto',
						'text-align': 'left'
					}
				)
			],
			className='input-wrapper'
		),
		html.H2(id='prob', style={
			'text-align': 'center'
		}),
		html.H2(id='predict_value', style={
			'text-align': 'center'
		}),
		html.Div(
			[dcc.Graph(id='predict_graph', figure=fig_2)],
			style={
				'width': '125%',
				'margin-right': 'auto',
				'margin-left': 'auto'
			}
		),
		html.Div(
			[dcc.Graph(id='bonus', figure=fig_2)],
			style={
				'width': '125%',
				'margin-right': 'auto',
				'margin-left': 'auto'
			}
		),
		dcc.Markdown('Created by Group Sichuan Hotpot ')
	],
	className='container',
	style={
		'width': '125%',
		'margin-right': 'auto',
		'margin-left': 'auto'
	}
)

app.css.append_css(
	{
		'external_url':
		'https://cdn.rawgit.com/gschivley/8040fc3c7e11d2a4e7f0589ffc829a02/raw/fe763af6be3fc79eca341b04cd641124de6f6f0d/dash.css'
	}
)


@app.callback(
	[
		dash.dependencies.Output('predict_graph', 'figure'),
		dash.dependencies.Output('predict_value', 'children'),
		dash.dependencies.Output('prob', 'children'),
	], [
		dash.dependencies.Input('age', 'value'),
		dash.dependencies.Input('sex', 'value'),
		dash.dependencies.Input('chest_pain_type', 'value'),
		dash.dependencies.Input('resting_blood_pressure', 'value'),
		dash.dependencies.Input('cholesterol', 'value'),
		dash.dependencies.Input('fasting_blood_sugar', 'value'),
		dash.dependencies.Input('rest_ecg', 'value'),
		dash.dependencies.Input('max_heart_rate_achieved', 'value'),
		dash.dependencies.Input('exercise_induced_angina', 'value'),
		dash.dependencies.Input('st_depression', 'value'),
		dash.dependencies.Input('st_slope', 'value'),
		dash.dependencies.Input('num_major_vessels', 'value'),
		dash.dependencies.Input('thalassemia', 'value')
	]
)
def update_figure(
	age, sex, chest_pain_type, resting_blood_pressure, cholesterol,
	fasting_blood_sugar, rest_ecg, max_heart_rate_achieved,
	exercise_induced_angina, st_depression, st_slope, num_major_vessels,
	thalassemia
):

	#	M = mm.MyModel()
	results = M.predict(
		[
			age, sex, chest_pain_type, resting_blood_pressure, cholesterol,
			fasting_blood_sugar, rest_ecg, max_heart_rate_achieved,
			exercise_induced_angina, st_depression, st_slope, num_major_vessels,
			thalassemia
		]
	)

	nmlize = lambda x: x * 1000 if x < 1 else (x * 100 if x < 10 else x)

	your_data = [
		nmlize(e)
		for e in [
			age, chest_pain_type, resting_blood_pressure, cholesterol,
			fasting_blood_sugar, rest_ecg, max_heart_rate_achieved,
			exercise_induced_angina, st_depression, st_slope, num_major_vessels,
			thalassemia
		]
	]

	filter_tags = [e for e in df.columns if e not in ['sex', 'target']]

	new_df = df[(df['sex'] == 1.0)] if sex else df[(df['sex'] == 0.0)]

	avg = go.Barpolar(
		r=[df[tag].astype('float64').mean() for tag in filter_tags],
		text=filter_tags,
		name='Male\'s average' if sex else 'Female\'s average',
		marker=dict(color='rgb(106,81,163)')
	)
	yours = go.Barpolar(
		r=your_data,
		text=filter_tags,
		name='Your figure',
		marker=dict(color='rgb(158,154,200)')
	)

	data = [avg, yours]
	layout = go.Layout(
		title='Your figure compairing with Male\' average'
		if sex else 'Your fingure compairing with Female\'s average',
		font=dict(size=16),
		legend=dict(font=dict(size=16)),
		radialaxis=dict(ticksuffix='%'),
		width=800,
		height=800,
		orientation=270
	)
	fig = go.Figure(data=data, layout=layout)

	prob = f'{int(100 * results[1])} percent possibility'
	res = '\t\tYou have heart disease.' if results[
		0
	] else '\t\tYou do not have heart disease.'

	return fig, res, prob


if __name__ == '__main__':
	app.run_server(debug=True)
