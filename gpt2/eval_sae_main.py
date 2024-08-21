from eval_gpt2 import *
from imports import *
from models import *
from ravel_data_prep import *

random.seed(2)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "mps"


def config(DEVICE):
    model = LanguageModel("openai-community/gpt2", device_map=DEVICE)
    intervened_token_idx = -8
    return model, intervened_token_idx


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--device", default="cuda:1")
    parser.add_argument("-met", "--method", required=True)
    parser.add_argument("-bs", "--batch_size", required=True)
    args = parser.parse_args()

    model, intervened_token_idx = config(args.device)

    model_sae_eval = eval_sae(
        model=model,
        DEVICE=args.device,
        method=args.method,
        intervened_token_idx=intervened_token_idx,
        batch_size=args.batch_size,
    )

    # TODO: Insert the tokenizaton and batching fucntion for the data
    with open("comfy_continent.json", "r") as f:
        contdata = json.load(f)

    with open("comfy_country.json", "r") as f1:
        countdata = json.load(f1)

    contsent = [sent[0] for sent in contdata]
    contlabel = [label[1] for label in contdata]

    countsent = [s[0] for s in countdata]
    countlabel = [l[1] for l in countdata]

    all_sent = contsent + countsent
    all_label = contlabel + countlabel

    model.tokenizer.padding_side = "left"

    t_sent = model.tokenizer(all_sent, return_tensors="pt", padding=True).to(
        args.device
    )
    t_label = model.tokenizer(all_label, return_tensors="pt", padding=True).to(
        args.device
    )

    # print(f"len of t_sent is {len(t_sent['input_ids'])}")
    # print(f"len of label is {len(t_label)}")
    # print(t_sent)
    indices = int(len(t_sent["input_ids"]) / 16)
    print(indices)
    (
        loss0_arr,
        loss1_arr,
        loss2_arr,
        loss3_arr,
        loss4_arr,
        loss5_arr,
        loss6_arr,
        loss7_arr,
        loss8_arr,
        loss9_arr,
        loss10_arr,
        loss11_arr,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [])

    with torch.no_grad():
        for i in tqdm(range(indices)):

            samples = t_sent["input_ids"][i : (i + 1) * 16]
            s_labels = t_label["input_ids"][i : (i + 1) * 16]

            (
                loss0,
                loss1,
                loss2,
                loss3,
                loss4,
                loss5,
                loss6,
                loss7,
                loss8,
                loss9,
                loss10,
                loss11,
            ) = model_sae_eval(samples)

            loss0_arr.append(loss0.mean(0).item())
            loss1_arr.append(loss1.mean(0).item())
            loss2_arr.append(loss2.mean(0).item())
            loss3_arr.append(loss3.mean(0).item())
            loss4_arr.append(loss4.mean(0).item())
            loss5_arr.append(loss5.mean(0).item())
            loss6_arr.append(loss6.mean(0).item())
            loss7_arr.append(loss7.mean(0).item())
            loss8_arr.append(loss8.mean(0).item())
            loss9_arr.append(loss9.mean(0).item())
            loss10_arr.append(loss10.mean(0).item())
            loss11_arr.append(loss11.mean(0).item())

            torch.cuda.empty_cache()

        mean0 = sum(loss0_arr) / len(loss0_arr)
        mean1 = sum(loss1_arr) / len(loss1_arr)
        mean2 = sum(loss2_arr) / len(loss2_arr)
        mean3 = sum(loss3_arr) / len(loss3_arr)
        mean4 = sum(loss4_arr) / len(loss4_arr)
        mean5 = sum(loss5_arr) / len(loss5_arr)
        mean6 = sum(loss6_arr) / len(loss6_arr)
        mean7 = sum(loss7_arr) / len(loss7_arr)
        mean8 = sum(loss8_arr) / len(loss8_arr)
        mean9 = sum(loss9_arr) / len(loss9_arr)
        mean10 = sum(loss10_arr) / len(loss10_arr)
        mean11 = sum(loss11_arr) / len(loss11_arr)

        print(mean0)
        print(mean1)
        print(mean2)
        print(mean3)
        print(mean4)
        print(mean5)
        print(mean6)
        print(mean7)
        print(mean8)
        print(mean9)
        print(mean10)
        print(mean11)
