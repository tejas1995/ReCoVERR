# Those are manual mapping that are not caught by our stemming rules or would
# would be done incorrectly by our automatic stemming rule. In details,
# the keys of the _MANUAL_MATCHES dict contains the original word and the value
# contains the transformation of the word expected by the OKVQA stemming rule.
# These manual rules were found by checking the `raw_answers` and the `answers`
# fields of the released OKVQA dataset and checking all things that were not
# properly mapped by our automatic rules. In particular some of the mapping
# are sometimes constant, e.g. christmas -> christmas which was incorrectly
# singularized by our inflection.singularize.
import re
import nltk
import spacy
from nltk.corpus.reader import VERB
import inflection
from utils.openai_utils import openai_caller
import string

_MANUAL_MATCHES = {
    "police": "police",
    "las": "las",
    "vegas": "vegas",
    "yes": "yes",
    "jeans": "jean",
    "hell's": "hell",
    "domino's": "domino",
    "morning": "morn",
    "clothes": "cloth",
    "are": "are",
    "riding": "ride",
    "leaves": "leaf",
    "dangerous": "danger",
    "clothing": "cloth",
    "texting": "text",
    "kiting": "kite",
    "firefighters": "firefight",
    "ties": "tie",
    "married": "married",
    "teething": "teeth",
    "gloves": "glove",
    "tennis": "tennis",
    "dining": "dine",
    "directions": "direct",
    "waves": "wave",
    "christmas": "christmas",
    "drives": "drive",
    "pudding": "pud",
    "coding": "code",
    "plating": "plate",
    "quantas": "quanta",
    "hornes": "horn",
    "graves": "grave",
    "mating": "mate",
    "paned": "pane",
    "alertness": "alert",
    "sunbathing": "sunbath",
    "tenning": "ten",
    "wetness": "wet",
    "urinating": "urine",
    "sickness": "sick",
    "braves": "brave",
    "firefighting": "firefight",
    "lenses": "lens",
    "reflections": "reflect",
    "backpackers": "backpack",
    "eatting": "eat",
    "designers": "design",
    "curiousity": "curious",
    "playfulness": "play",
    "blindness": "blind",
    "hawke": "hawk",
    "tomatoe": "tomato",
    "rodeoing": "rodeo",
    "brightness": "bright",
    "circuses": "circus",
    "skateboarders": "skateboard",
    "staring": "stare",
    "electronics": "electron",
    "electicity": "elect",
    "mountainous": "mountain",
    "socializing": "social",
    "hamburgers": "hamburg",
    "caves": "cave",
    "transitions": "transit",
    "wading": "wade",
    "creame": "cream",
    "toileting": "toilet",
    "sautee": "saute",
    "buildings": "build",
    "belongings": "belong",
    "stockings": "stock",
    "walle": "wall",
    "cumulis": "cumuli",
    "travelers": "travel",
    "conducter": "conduct",
    "browsing": "brows",
    "pooping": "poop",
    "haircutting": "haircut",
    "toppings": "top",
    "hearding": "heard",
    "sunblocker": "sunblock",
    "bases": "base",
    "markings": "mark",
    "mopeds": "mope",
    "kindergartener": "kindergarten",
    "pies": "pie",
    "scrapbooking": "scrapbook",
    "couponing": "coupon",
    "meetings": "meet",
    "elevators": "elev",
    "lowes": "low",
    "men's": "men",
    "childrens": "children",
    "shelves": "shelve",
    "paintings": "paint",
    "raines": "rain",
    "paring": "pare",
    "expressions": "express",
    "routes": "rout",
    "pease": "peas",
    "vastness": "vast",
    "awning": "awn",
    "boy's": "boy",
    "drunkenness": "drunken",
    "teasing": "teas",
    "conferences": "confer",
    "ripeness": "ripe",
    "suspenders": "suspend",
    "earnings": "earn",
    "reporters": "report",
    "kid's": "kid",
    "containers": "contain",
    "corgie": "corgi",
    "porche": "porch",
    "microwaves": "microwave",
    "batter's": "batter",
    "sadness": "sad",
    "apartments": "apart",
    "oxygenize": "oxygen",
    "striping": "stripe",
    "purring": "pure",
    "professionals": "profession",
    "piping": "pipe",
    "farmer's": "farmer",
    "potatoe": "potato",
    "emirates": "emir",
    "womens": "women",
    "veteran's": "veteran",
    "wilderness": "wilder",
    "propellers": "propel",
    "alpes": "alp",
    "charioteering": "chariot",
    "swining": "swine",
    "illness": "ill",
    "crepte": "crept",
    "adhesives": "adhesive",
    "regent's": "regent",
    "decorations": "decor",
    "rabbies": "rabbi",
    "overseas": "oversea",
    "travellers": "travel",
    "casings": "case",
    "smugness": "smug",
    "doves": "dove",
    "nationals": "nation",
    "mustange": "mustang",
    "ringe": "ring",
    "gondoliere": "gondolier",
    "vacationing": "vacate",
    "reminders": "remind",
    "baldness": "bald",
    "settings": "set",
    "glaced": "glace",
    "coniferous": "conifer",
    "revelations": "revel",
    "personals": "person",
    "daughter's": "daughter",
    "badness": "bad",
    "projections": "project",
    "polarizing": "polar",
    "vandalizers": "vandal",
    "minerals": "miner",
    "protesters": "protest",
    "controllers": "control",
    "weddings": "wed",
    "sometimes": "sometime",
    "earing": "ear",
}


class OKVQAStemmer:
    """Stemmer to match OKVQA v1.1 procedure."""

    def __init__(self):
        self._wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()

    def stem(self, input_string):
        """Apply stemming."""
        word_and_pos = nltk.pos_tag(nltk.tokenize.word_tokenize(input_string))
        stemmed_words = []
        for w, p in word_and_pos:
            if w in _MANUAL_MATCHES:
                w = _MANUAL_MATCHES[w]
            elif w.endswith("ing"):
                w = self._wordnet_lemmatizer.lemmatize(w, VERB)
            elif p.startswith("NNS") or p.startswith("NNPS"):
                w = inflection.singularize(w)
            stemmed_words.append(w)
        return " ".join(stemmed_words)


stemmer = OKVQAStemmer()


def postprocess_ok_vqa_generation(predictions) -> str:
    prediction = re.split("Question|Answer", predictions, 1)[0]
    prediction_stem = stemmer.stem(prediction)
    return prediction_stem

lemmatizer = spacy.load("en_core_web_sm")
def lemmatize(answer):
    doc = lemmatizer(answer)

    words = []
    for token in doc:
        if token.pos_ in ["NOUN", "VERB"]:
            words.append(token.lemma_)
        else:
            words.append(token.text)
    answer = " ".join(words)
    return answer



def get_closest_match_option(vqa_question, predicted_answer, choices):
    
    prompt = f"Question: {vqa_question}\n" \
        f"Options: {', '.join(choices)}\n" \
        f"Predicted answer: {predicted_answer}\n\n" \
        "Does the predicted answer identical to any of the options, or have the same meaning as any of options? " \
        "If yes, say \"answer:\" followed by the option with the same meaning. " \
        "Do not select more than one option. " \
        "If none of the choices mean the same thing as the predicted answer, say no."
    message = [{"role": "user", "content": prompt}]
    response = openai_caller(message, model='gpt4', max_new_tokens=10, temperature=0.0)
    response = response.lower()
    if response.startswith('answer:'):
        response = response[7:]
    #print(prompt, response)
    response = response.strip().strip(string.punctuation)
    if response == 'no':
        return None
    else:
        return response

def get_closest_match_annotatedanswer(vqa_question, predicted_answer, score_dict):
    prompt = f"Question: {vqa_question}\n" \
        f"Annotated answers: {', '.join([k for k, v in score_dict.items() if v > 0])}\n" \
        f"Predicted answer: {predicted_answer}\n\n" \
        "Does the predicted answer mean the same as any of the annotated answers? " \
        "If yes, say which annotated answer means the predicted answer, preceded by \"answer:\". " \
        "Do not select more than one annotated answer. " \
        "If none of the annotated answers mean the same thing as the predicted answer, say no."

    messages = [{'role': 'user', 'content': prompt}]
    response = openai_caller(messages, model='gpt4', max_new_tokens=10, temperature=0.0)
    response = response.lower()
    if response.startswith('answer:'):
        response = response[7:]
    response = response.strip().split(',')[0].strip().strip(string.punctuation)
    if response == 'no':
        return None
    else:
        return response