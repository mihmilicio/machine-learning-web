
from flask import render_template

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import base64
from io import BytesIO

import warnings
warnings.filterwarnings('always')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from iris_ml import app
from iris_ml.forms import FormClassifiers, FormKNNParameter, FormMLPParameter, FormDTParameter, FormRFParameter



def get_classifier_form(classifier):
    match classifier:
        case 'knn':
          return FormKNNParameter()
        case 'mlp':
          return FormMLPParameter()
        case 'dt':
          return FormDTParameter()
        case 'rf':
          return FormRFParameter()

def get_classifier_instance(classifier, form):
  match classifier:
    case 'knn':
      return KNeighborsClassifier(n_neighbors=form.n_neighbors.data)
    case 'mlp':
      return MLPClassifier(random_state=form.random_state.data, max_iter=form.max_iter.data)
    case 'dt':
      return DecisionTreeClassifier(max_leaf_nodes=form.max_leaf_nodes.data, random_state=form.random_state.data)
    case 'rf':
      return RandomForestClassifier(max_depth=form.max_depth.data, random_state=form.random_state.data)


@app.route("/", methods=['POST', 'GET'])
@app.route('/iris', methods=['POST', 'GET'])
def iris():
  form = FormClassifiers()

  if form.validate_on_submit():
    classifier = form.classifier.data
    
    formClassifier = get_classifier_form(classifier)

    # prepara dados
    iris = load_iris()
    X = iris.data # caracteristica
    y = iris.target # rotulos
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # ml
    clf = get_classifier_instance(classifier, formClassifier)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # métricas
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    macro = f1_score(y_test, y_pred, average='macro')

    # matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    classes = iris.target_names.tolist()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    disp.plot()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    img_src = f'data:image/png;base64,{data}'
    
    # tela com previsões
    return render_template("iris.html", 
                          form=formClassifier, 
                          classifier=classifier, 
                          chart=img_src,
                          accuracy=f'{acc:.3f}',
                          macro_avg=f'{macro:.3f}')
  
  # tela inicial
  return render_template("iris.html", form=form)
