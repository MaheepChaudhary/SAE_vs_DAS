from imports import *

# making the graph for continent


def graph(
    neuron_masking,
    das_masking,
    sae_neel,
    sae_openai,
    sae_apollo_e2eds,
    sae_apollo,
    method,
):

    data = {}
    data = {
        "Layer": list(range(12)),
        "neuron_masking": neuron_masking,
        "das_masking": das_masking,
        "sae_neel": sae_neel,
        "sae_openai": sae_openai,
        "sae_apollo_e2eds": sae_apollo_e2eds,
        "sae_apollo": sae_apollo,
    }

    df = pd.DataFrame(data)
    df.plot(
        x="Layer",
        y=[
            "neuron_masking",
            "das_masking",
            "sae_neel",
            "sae_openai",
            "sae_apollo_e2eds",
            "sae_apollo",
        ],
    )

    plt.title(f"Accuracy for {method}")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


neuron_masking_continent = [
    0.72,
    0.72,
    0.73,
    0.71,
    0.73,
    0.72,
    0.72,
    0.74,
    0.75,
    0.72,
    0.69,
    0.68,
]
das_masking_continent = [
    0.94,
    0.96,
    0.92,
    0.95,
    0.97,
    0.96,
    0.96,
    0.95,
    0.89,
    0.81,
    0.77,
    0.68,
]
sae_neel_masking_continent = [
    0.63,
    0.45,
    0.4,
    0.53,
    0.6,
    0.62,
    0.6,
    0.64,
    0.67,
    0.52,
    0.21,
    0.68,
]
sae_openai_masking_continent = [
    0.72,
    0.73,
    0.7,
    0.67,
    0.69,
    0.70,
    0.64,
    0.61,
    0.66,
    0.68,
    0.67,
    0.68,
]
sae_apollo_e2eds_masking_continent = [0, 0.57, 0, 0, 0, 0.39, 0, 0, 0, 0.64, 0, 0]
sae_apollo_masking_continent = [0, 0.53, 0, 0, 0, 0.39, 0, 0, 0, 0.72, 0, 0]

neuron_masking_country = [
    0.74,
    0.71,
    0.74,
    0.73,
    0.73,
    0.73,
    0.71,
    0.73,
    0.54,
    0.55,
    0.58,
    0.56,
]
das_masking_country = [
    0.93,
    0.94,
    0.91,
    0.94,
    0.95,
    0.94,
    0.95,
    0.94,
    0.74,
    0.70,
    0.68,
    0.56,
]
sae_neel_masking_country = [
    0.64,
    0.42,
    0.38,
    0.57,
    0.65,
    0.64,
    0.63,
    0.65,
    0.60,
    0.49,
    0.22,
    0.56,
]
sae_openai_masking_country = [
    0.71,
    0.73,
    0.71,
    0.67,
    0.71,
    0.7,
    0.63,
    0.59,
    0.42,
    0.51,
    0.55,
    0.56,
]
sae_apollo_e2eds_masking_country = [0, 0.59, 0, 0, 0, 0.44, 0, 0, 0, 0.6, 0, 0]
sae_apollo_masking_country = [0, 0.54, 0, 0, 0, 0.44, 0, 0, 0, 0.57, 0, 0]


print(len(neuron_masking_continent))
print(len(das_masking_continent))
print(len(sae_neel_masking_continent))
print(len(sae_openai_masking_continent))
print(len(sae_apollo_e2eds_masking_continent))
print(len(sae_apollo_masking_continent))


graph(
    neuron_masking_continent,
    das_masking_continent,
    sae_neel_masking_continent,
    sae_openai_masking_continent,
    sae_apollo_e2eds_masking_continent,
    sae_apollo_masking_continent,
    "continent",
)

graph(
    neuron_masking_country,
    das_masking_country,
    sae_neel_masking_country,
    sae_openai_masking_country,
    sae_apollo_e2eds_masking_country,
    sae_apollo_masking_country,
    "country",
)
