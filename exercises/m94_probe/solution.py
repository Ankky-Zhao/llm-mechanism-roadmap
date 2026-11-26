import warnings
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 160 cities, 40 per label. The labels are the task of the probe.
CITIES = {
    "Europe": [
        "London", "Paris", "Berlin", "Madrid", "Rome", "Vienna", "Prague", "Warsaw",
        "Budapest", "Amsterdam", "Brussels", "Lisbon", "Dublin", "Athens", "Stockholm", "Oslo",
        "Copenhagen", "Helsinki", "Zurich", "Munich", "Hamburg", "Milan", "Naples", "Barcelona",
        "Lyon", "Marseille", "Manchester", "Edinburgh", "Glasgow", "Geneva", "Frankfurt", "Venice",
        "Florence", "Seville", "Porto", "Krakow", "Bucharest", "Sofia", "Belgrade", "Zagreb",
    ],
    "Asia": [
        "Tokyo", "Beijing", "Shanghai", "Seoul", "Bangkok", "Jakarta", "Manila", "Hanoi",
        "Singapore", "Mumbai", "Delhi", "Kolkata", "Chennai", "Bangalore", "Karachi", "Lahore",
        "Dhaka", "Kathmandu", "Colombo", "Tehran", "Baghdad", "Riyadh", "Dubai", "Doha",
        "Kabul", "Tashkent", "Osaka", "Kyoto", "Guangzhou", "Shenzhen", "Wuhan", "Chengdu",
        "Taipei", "Busan", "Yangon", "Islamabad", "Hyderabad", "Nanjing", "Tianjin", "Pyongyang",
    ],
    "Africa": [
        "Cairo", "Lagos", "Nairobi", "Johannesburg", "Accra", "Dakar", "Casablanca", "Tunis",
        "Algiers", "Khartoum", "Kampala", "Kigali", "Luanda", "Maputo", "Harare", "Lusaka",
        "Kinshasa", "Abuja", "Tripoli", "Mogadishu", "Durban", "Pretoria", "Alexandria", "Marrakech",
        "Bamako", "Freetown", "Monrovia", "Windhoek", "Gaborone", "Abidjan", "Kano", "Mombasa",
        "Rabat", "Niamey", "Lilongwe", "Antananarivo", "Ouagadougou", "Conakry", "Asmara", "Djibouti",
    ],
    "Americas": [
        "Chicago", "Boston", "Houston", "Dallas", "Seattle", "Denver", "Miami", "Atlanta",
        "Detroit", "Philadelphia", "Toronto", "Montreal", "Vancouver", "Ottawa", "Havana", "Bogota",
        "Lima", "Santiago", "Caracas", "Quito", "Montevideo", "Brasilia", "Guadalajara", "Monterrey",
        "Phoenix", "Portland", "Calgary", "Winnipeg", "Medellin", "Recife", "Salvador", "Curitiba",
        "Valparaiso", "Cusco", "Tijuana", "Pittsburgh", "Baltimore", "Minneapolis", "Nashville", "Honolulu",
    ],
}
CONTINENTS = list(CITIES)


def city_dataset():
    """Returns (names, y): the 160 city names and their labels 0..3 as a numpy array."""
    names = [c for k in CONTINENTS for c in CITIES[k]]
    y = np.array([i for i, k in enumerate(CONTINENTS) for _ in CITIES[k]])
    return names, y


def load_models():
    """GPT-2 small, its tokenizer, and a GPT-2 small with random weights as a control."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    torch.manual_seed(0)
    random_model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("gpt2")).eval()
    return model, tok, random_model


@torch.no_grad()
def last_token_states(model, tok, texts):
    """Hidden states at the last token of every text. Returns a numpy array (N, 13, 768).

    Index 0 is the embedding, index l the state after block l - 1. In the GPT-2 implementation of
    transformers the last index has the final LayerNorm applied, which does not matter for a probe.
    """
    rows = []
    for t in texts:
        hs = model(tok(t, return_tensors="pt").input_ids, output_hidden_states=True).hidden_states
        rows.append(torch.stack([h[0, -1] for h in hs]))
    return torch.stack(rows).numpy()


def make_probe(C):
    """Standardize the features, then multinomial logistic regression with L2 penalty 1 / C."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def probe_cv_accuracy(X, y, C=0.1, folds=5, seed=0):
    """Mean accuracy on held-out examples over stratified k-fold cross-validation. X (N, d), y (N,)."""
    accs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
            probe = make_probe(C).fit(X[train], y[train])
            accs.append(probe.score(X[test], y[test]))
    return float(np.mean(accs))


def probe_train_accuracy(X, y, C=0.1):
    """Accuracy on the same examples the probe was trained on."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(make_probe(C).fit(X, y).score(X, y))
