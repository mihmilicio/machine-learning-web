from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, IntegerField
from wtforms.validators import DataRequired, NumberRange

class FormClassifiers(FlaskForm):
    classifier = SelectField('Classifier', 
                            validators=[DataRequired()], 
                            coerce=str, 
                            choices=[('knn', 'KNeighborsClassifier'), ('mlp', 'MLPClassifier'), ('dt', 'DecisionTreeClassifier'), ('rf', 'RandomForestClassifier')])
    btn = SubmitField('Select Classifier')
    classify_btn = SubmitField('Classify')

class FormKNNParameter(FormClassifiers):
    n_neighbors = IntegerField('n_neighbors', validators=[DataRequired()], default=3)

class FormMLPParameter(FormClassifiers):
    random_state = IntegerField('random_state', validators=[DataRequired()], default=1)
    max_iter = IntegerField('max_iter', validators=[DataRequired()], default=300)

class FormDTParameter(FormClassifiers):
    random_state = IntegerField('random_state', validators=[DataRequired()], default=0)
    max_leaf_nodes = IntegerField('max_leaf_nodes', validators=[DataRequired(), NumberRange(min=2)], default=4)

class FormRFParameter(FormClassifiers):
    random_state = IntegerField('random_state', validators=[DataRequired()], default=0)
    max_depth = IntegerField('max_depth', validators=[DataRequired()], default=2)
