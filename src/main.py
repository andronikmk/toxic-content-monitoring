from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from src.model import ToxicClassifierModel, vectors
from src.preprocessor import normalize_comment, clean_text

# load model
model = ToxicClassifierModel()
model.load_state_dict(torch.load("data/TCM_2.pt", map_location=torch.device('cpu')))
model.eval()

# FastAPI
app = FastAPI()

# makes request body rather than parameter
class Data(BaseModel):
    text: str

@app.post("/predict/")
async def predict(data: Data):
    """
    This API accepts string based requests and preprocesses the text before
    running it through the predictive model.

    **Preprocessing:**
    The following are removed:
    - Newline characters
    - Return characters
    - Leading and trailing white spaces
    - Usernames if indicated with the term 'User'
    - IP addresses and user IDs
    - Non-printable characters e.g. unicode

    Words that are disguised using characters such as * or @ are replaced with
    letters and common astericks words are replaced with coordinating word.

    The API also preprocesses text so that if toxic words are mixed in with
    non-toxic content, it will split the text before and after so the toxic words
    are identified.

    **Making Requests:**

    Example Request Body:

    {"text": "string"}

    **Model:**
    
    The model is used to determine if the text contains toxic or offensive content.

    The possible labels are:
    - toxic
    - severe toxic
    - obscene
    - threat
    - insult
    - identity hate

    The API returns the cleaned text, all labels, True of False for each label,
    and the predicted probability of each.

    The current model is a Bi-directional LSTM + GRU neural network made with
    PyTorch, assuming FastText vectorization. Considerable preprocessing is
    performed on the text before vectorization. The metrics used in evaluating
    this model are F1 and ROC-AUC scores.

    F1 score is defined as the harmonic mean between precision and recall on a scale of
    0 to 1. Recall demonstrates how effectively this model identifies all relevant
    instances of toxicity. Precision demonstrates how effectively the model returns
    only these relevant instances.

    The AUC score represents the measure of separability, in this case, distinguishing
    between toxic and non-toxic content. Also on a scale of 0 to 1, a high AUC score
    indicates the model successfully classifies toxic vs non-toxic. The ROC represents
    the probability curve.

    The F1 score for this model is 0.753 and the ROC-AUC score is 0.987.
    """

    # clean text to remove characters and metadata that may interfere with accuracy of model
    text = clean_text(data.text)

    # normalize comment (see preprocessor.py file)
    normalized_words = normalize_comment(text)

    # vectorize the normalized comment using FastText
    var_vectorized_sentence = Variable(torch.nn.utils.rnn.pad_sequence([vectors[normalized_words]]).permute(1,0,2)).float()

    # creates predictions for text on different labels
    output = model(var_vectorized_sentence)

    # removes prediction from tensor object and into a list
    predProb = output.data.numpy().tolist()[0]

    # threshold can be changed in order to increase positive recall scores
    threshold = 0.5

    # returns true or false for each class based on prediction
    preds = [ p >= threshold for p in predProb ]

    # identify labels to process predictions
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # returns for each class: boolean classification and predicted probability
    display_results = {label:{'prediction':pred, 'probability':round(proba, 5)} for label,pred,proba in zip(labels, preds, predProb)}

    # returns cleaned input text and display_results
    return {"text": text, "results": display_results}


# Customize documentation
def custom_openapi():
    """[Generate the OpenAPI schema]

    Returns:
        [function]: [Use the same utility function to generate the OpenAPI schema, inside a custom_openapi() function]
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Toxicity Classification API",
        version="2.0.5",
        description="This API is used to detect toxicity of varying degrees in text. "\
                    "To read the documentation for post request click the **POST** button. "\
                    "Clicking the **Try it out** button will allow you to test requests."  ,
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi