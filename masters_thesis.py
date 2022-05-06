# Authors: Alexander Dolk and Hjalmar Davidsen
import torch
import pandas as pd
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          PreTrainedModel, BertModel, BertForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, accuracy_score
from statistics import mean
from sklearn.model_selection import KFold
import operator
import matplotlib.pyplot as plt


class GastroDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class BertForMultilabelSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.float().view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)

def get_dictionary_from_df(df, names):
    dictionary = {}
    for col in names:
        dictionary[col] = (df[col] == 1).sum()

    sorted_dict = dict(sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True))
    return sorted_dict

def display_dictionary(dictionary):
    sorted_list_of_tuples = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    x, y = zip(*sorted_list_of_tuples)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.xticks(rotation=90)
    plt.show()

def visualise_ICD_code_distribution():
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    data = pd.read_csv(
        '<Filepath to data with discharge summaries (no duplicates)>',
        delimiter=',')

    # Look at distribution of full dataframe
    full_names = 'K029,K048,K116,K117,K209,K210,K219,K220,K221,K222,K223,K224,K225,K226,K227,K228,K229,K250,K251,K252,K253,K254,K257,K259,K260,K261,K262,K263,K267,K269,K270,K271,K272,K274,K277,K279,K281,K289,K290,K291,K293,K294,K295,K296,K297,K298,K299,K309,K310,K311,K315,K316,K317,K318,K319,K350,K351,K352,K353,K358,K359,K369,K379,K388,K402,K402A,K403,K404,K409,K409A,K409B,K410,K413,K413B,K414,K419,K420,K421,K429,K430,K430A,K430B,K431,K432,K432A,K432B,K433,K434,K435,K435A,K435B,K436,K439,K439A,K439B,K440,K441,K449,K450,K458,K460,K469,K500,K501,K508,K509,K510,K512,K513,K515,K518,K519,K520,K522,K522A,K523,K528,K529,K529A,K529W,K550,K551,K552,K558,K559,K560,K561,K562,K563,K564,K565,K566,K567,K570,K571,K572,K573,K574,K575,K578,K579,K580,K589,K590,K591,K592,K593,K594,K598,K599,K600,K601,K602,K603,K604,K605,K610,K611,K612,K613,K620,K621,K622,K623,K624,K625,K626,K627,K628,K629,K630,K631,K632,K633,K634,K635,K638,K639,K642,K643,K648,K649,K650,K658,K659,K660,K661,K668,K700,K701,K703,K704,K709,K712,K720,K729,K732,K738,K739,K740,K743,K745,K746,K750,K751,K754,K758,K760,K766,K767,K768,K769,K800,K801,K802,K803,K804,K805,K808,K810,K811,K818,K819,K820,K822,K823,K824,K828,K830,K830A,K830X,K831,K833,K838,K839,K850,K851,K852,K853,K858,K859,K860,K861,K862,K863,K868,K869,K900,K900A,K900B,K900X,K904,K908,K910,K911,K912,K913,K914,K915,K918,K920,K921,K922'
    full_names = full_names.split(",")
    data_only_labels = data.drop(['patientnr'], axis=1)
    data_only_labels = data_only_labels.drop(['anteckning'], axis=1)

    sorted_dict = get_dictionary_from_df(data_only_labels, full_names)
    print("sorted_dict:")
    print(sorted_dict)
    display_dictionary(sorted_dict)

    dict_with_values_over_hundred = dict((k, v) for k, v in sorted_dict.items() if v >= 100)
    print("dict_with_values_over_hundred:")
    print(dict_with_values_over_hundred)
    display_dictionary(dict_with_values_over_hundred)

def create_subset_of_discharge_summaries():
    #keys = list(dict_with_values_over_hundred.keys())
    #print("keys:")
    #print(keys)
    #print(data.shape)

    data = pd.read_csv(
        '<Filepath to data with discharge summaries (no duplicates)>',
        delimiter=',')

    d = data.loc[((data['K567'] == 1) | (data['K573'] == 1) | (data['K358'] == 1) | (data['K590'] == 1) |
                  (data['K800'] == 1) | (data['K379'] == 1) | (data['K802'] == 1) | (data['K610'] == 1) |
                  (data['K566'] == 1) | (data['K509'] == 1) | (data['K859'] == 1) | (data['K572'] == 1) |
                  (data['K353'] == 1) | (data['K650'] == 1) | (data['K922'] == 1) | (data['K565'] == 1) |
                  (data['K210'] == 1) | (data['K560'] == 1))]
    print(d.shape)

    # show distribution of d
    d_only_labels = d.drop(['patientnr'], axis=1)
    d_only_labels = d_only_labels.drop(['anteckning'], axis=1)
    print(d_only_labels)
    full_names = 'K029,K048,K116,K117,K209,K210,K219,K220,K221,K222,K223,K224,K225,K226,K227,K228,K229,K250,K251,K252,K253,K254,K257,K259,K260,K261,K262,K263,K267,K269,K270,K271,K272,K274,K277,K279,K281,K289,K290,K291,K293,K294,K295,K296,K297,K298,K299,K309,K310,K311,K315,K316,K317,K318,K319,K350,K351,K352,K353,K358,K359,K369,K379,K388,K402,K402A,K403,K404,K409,K409A,K409B,K410,K413,K413B,K414,K419,K420,K421,K429,K430,K430A,K430B,K431,K432,K432A,K432B,K433,K434,K435,K435A,K435B,K436,K439,K439A,K439B,K440,K441,K449,K450,K458,K460,K469,K500,K501,K508,K509,K510,K512,K513,K515,K518,K519,K520,K522,K522A,K523,K528,K529,K529A,K529W,K550,K551,K552,K558,K559,K560,K561,K562,K563,K564,K565,K566,K567,K570,K571,K572,K573,K574,K575,K578,K579,K580,K589,K590,K591,K592,K593,K594,K598,K599,K600,K601,K602,K603,K604,K605,K610,K611,K612,K613,K620,K621,K622,K623,K624,K625,K626,K627,K628,K629,K630,K631,K632,K633,K634,K635,K638,K639,K642,K643,K648,K649,K650,K658,K659,K660,K661,K668,K700,K701,K703,K704,K709,K712,K720,K729,K732,K738,K739,K740,K743,K745,K746,K750,K751,K754,K758,K760,K766,K767,K768,K769,K800,K801,K802,K803,K804,K805,K808,K810,K811,K818,K819,K820,K822,K823,K824,K828,K830,K830A,K830X,K831,K833,K838,K839,K850,K851,K852,K853,K858,K859,K860,K861,K862,K863,K868,K869,K900,K900A,K900B,K900X,K904,K908,K910,K911,K912,K913,K914,K915,K918,K920,K921,K922'
    full_names = full_names.split(",")
    d_dict = get_dictionary_from_df(d_only_labels, full_names)
    print("d_dict:")
    print(d_dict)

    final_table_columns = ['patientnr', 'anteckning', 'K567', 'K573', 'K358', 'K590', 'K800', 'K379', 'K802', 'K610',
                           'K566', 'K509', 'K859',
                           'K572', 'K353', 'K650', 'K922', 'K565', 'K210', 'K560']
    d1 = d[d.columns.intersection(final_table_columns)]

    d1.to_csv(
        '<Filepath to subset of data with discharge summaries>',
        index=False)

def create_training_validation_and_test_sets():
    data = pd.read_csv('<Filepath to subset of data with discharge summaries>')
    data_without_test, test = train_test_split(data, test_size=0.1, random_state=42)
    train, validation = train_test_split(data_without_test, test_size=0.2, random_state=42)

    data_without_test.to_csv('<Filepath to data without test set>', index=False)
    train.to_csv('<Filepath to train data>', index=False)
    validation.to_csv('<Filepath to validation data>', index=False)
    test.to_csv('<Filepath to test data>', index=False)

def cross_validation():
    filepath_train = '<Filepath to data without test set>'
    data_train = pd.read_csv(filepath_train, delimiter=',')
    label_cols = [c for c in data_train.columns if c not in ["patientnr", "anteckning"]]
    data_train["labels"] = data_train[label_cols].values.tolist()

    tokenizer = AutoTokenizer.from_pretrained(
        '<Filepath to sweDeClin-BERT tokenizer>',
        local_files_only=True,
        model_max_length=512,
        max_len=512,
        truncation=True,
        padding='Longest'
        )

    # prepare cross validation
    n = 5
    kf = KFold(n_splits=n, random_state=42, shuffle=True)
    iteration = 0
    for train_index, val_index in kf.split(data_train):
        iteration = iteration + 1
        print("Iteration", iteration)
        # splitting Dataframe (dataset not included)
        train_df = data_train.iloc[train_index]
        val_df = data_train.iloc[val_index]

        train_encodings = tokenizer(train_df["anteckning"].values.tolist(), truncation=True)
        val_encodings = tokenizer(val_df["anteckning"].values.tolist(), truncation=True)

        train_labels = train_df["labels"].values.tolist()
        val_labels = val_df["labels"].values.tolist()

        val_labels_as_floats = []
        for list in val_labels:
            new_inner_list = []
            for number in list:
                new_float = float(number)
                new_inner_list.append(new_float)
            val_labels_as_floats.append(new_inner_list)

        train_dataset = GastroDataset(train_encodings, train_labels)
        val_dataset = GastroDataset(val_encodings, val_labels_as_floats)

        num_labels = 18
        model = BertForMultilabelSequenceClassification.from_pretrained(
            '<Filepath to sweDeClin-BERT model>',
            local_files_only=True, problem_type="multi_label_classification", num_labels=num_labels).to('cuda')

        batch_size = 2
        logging_steps = len(train_dataset) // batch_size

        args = TrainingArguments(
            output_dir="output_directory",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            gradient_accumulation_steps=16,
            warmup_steps=155,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=10,
            weight_decay=0.01,
            logging_steps=logging_steps
        )

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer)

        metrics = trainer.evaluate()
        print(metrics)

        trainer.train()
        metrics2 = trainer.evaluate()
        print(metrics2)

def fine_tune_model():
    filepath_train = '<Filepath to train data>'
    data_train = pd.read_csv(filepath_train, delimiter=',')
    label_cols = [c for c in data_train.columns if c not in ["patientnr", "anteckning"]]
    data_train["labels"] = data_train[label_cols].values.tolist()
    filepath_validation = '<Filepath to validation data>'
    data_validation = pd.read_csv(filepath_validation, delimiter=',')
    label_cols_val = [c for c in data_validation.columns if c not in ["patientnr", "anteckning"]]
    data_validation["labels"] = data_validation[label_cols_val].values.tolist()

    tokenizer = AutoTokenizer.from_pretrained(
        '<Filepath to sweDeClin-BERT tokenizer>',
        local_files_only=True,
        model_max_length=512,
        max_len=512,
        truncation=True,
        padding='Longest'
        )

    train_encodings = tokenizer(data_train["anteckning"].values.tolist(), truncation=True)
    val_encodings = tokenizer(data_validation["anteckning"].values.tolist(), truncation=True)

    train_labels = data_train["labels"].values.tolist()
    val_labels = data_validation["labels"].values.tolist()

    test_labels_as_floats = []
    for list in val_labels:
        new_inner_list = []
        for number in list:
            new_float = float(number)
            new_inner_list.append(new_float)
        test_labels_as_floats.append(new_inner_list)

    train_dataset = GastroDataset(train_encodings, train_labels)
    val_dataset = GastroDataset(val_encodings, test_labels_as_floats)

    num_labels = 18
    model = BertForMultilabelSequenceClassification.from_pretrained(
        '<Filepath to sweDeClin-BERT model>',
        local_files_only=True, problem_type="multi_label_classification", num_labels=num_labels).to('cuda')

    batch_size = 2
    logging_steps = len(train_dataset) // batch_size

    args = TrainingArguments(
        output_dir="output_directory",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        gradient_accumulation_steps=16,
        warmup_steps=155,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=9,
        weight_decay=0.01,
        logging_steps=logging_steps
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer)

    metrics = trainer.evaluate()
    print(metrics)

    trainer.train()
    metrics2 = trainer.evaluate()
    print(metrics2)

    model_dir = '<Filepath to where you want to save model and tokenizer>'
    model.save_pretrained(model_dir + '<name of model>')
    tokenizer.save_pretrained(model_dir + '<name of tokenizer>')

def test_model():
    tokenizer = AutoTokenizer.from_pretrained(
        '<Filepath to fine-tuned tokenizer>',
        local_files_only=True,
        model_max_length=512,
        max_len=512,
        truncation=True,
        padding='Longest'
        )

    model1 = AutoModelForSequenceClassification.from_pretrained(
        '<Filepath to fine-tuned model>', local_files_only=True,
        problem_type="multi_label_classification", num_labels=18)

    test_data = pd.read_csv('<Filepath to fine-tuned model>')

    label_cols_test = [c for c in test_data.columns if c not in ["patientnr", "anteckning"]]

    test_data["labels"] = test_data[label_cols_test].values.tolist()

    true_values = []
    predictions = []

    for index, row in test_data.iterrows():
        anteckning = row['anteckning']
        correct_labels = row['labels']
        predictions_as_probs = get_prediction(anteckning, model1, tokenizer)
        pred = calc_threshold(predictions_as_probs)
        true_values.append(correct_labels)
        predictions.append(pred)

    acc = []

    prec_we = []
    rec_we = []
    f1_we = []

    prec_mic = []
    rec_mic = []
    f1_mic = []

    prec_mac = []
    rec_mac = []
    f1_mac = []

    for true_val, pred in zip(true_values, predictions):
        acc.append(accuracy_score(y_true=true_val, y_pred=pred))

        prec_we.append(precision_score(y_true=true_val, y_pred=pred, average='weighted'))
        rec_we.append(recall_score(y_true=true_val, y_pred=pred, average='weighted'))
        f1_we.append(f1_score(y_true=true_val, y_pred=pred, average='weighted'))

        prec_mic.append(precision_score(y_true=true_val, y_pred=pred, average='micro'))
        rec_mic.append(recall_score(y_true=true_val, y_pred=pred, average='micro'))
        f1_mic.append(f1_score(y_true=true_val, y_pred=pred, average='micro'))

        prec_mac.append(precision_score(y_true=true_val, y_pred=pred, average='macro'))
        rec_mac.append(recall_score(y_true=true_val, y_pred=pred, average='macro'))
        f1_mac.append(f1_score(y_true=true_val, y_pred=pred, average='macro'))

    # Calculate mean acc, prec, rec, f1 from the lists and print it
    print('mean(acc):')
    print(mean(acc))

    print('mean(prec_we):')
    print(mean(prec_we))
    print('mean(rec_we):')
    print(mean(rec_we))
    print('mean(f1_we):')
    print(mean(f1_we))

    print('mean(prec_mic):')
    print(mean(prec_mic))
    print('mean(rec_mic):')
    print(mean(rec_mic))
    print('mean(f1_mic):')
    print(mean(f1_mic))

    print('mean(prec_mac):')
    print(mean(prec_mac))
    print('mean(rec_mac):')
    print(mean(rec_mac))
    print('mean(f1_mac):')
    print(mean(f1_mac))

def get_prediction(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return probs


def calc_threshold(predictions_as_probs):
    list_of_predictions_as_zeroes_and_ones = []
    a = []
    a = predictions_as_probs[0]
    listus = a.tolist()
    for value in listus:
        if value > 0.5:
            list_of_predictions_as_zeroes_and_ones.append(1)
        else:
            list_of_predictions_as_zeroes_and_ones.append(0)

    return list_of_predictions_as_zeroes_and_ones

#Remove filepaths before uploading
def get_prediction_for_single_discharge_summary(discharge_summary):
    tokenizer = AutoTokenizer.from_pretrained(
        '<Filepath to fine-tuned tokenizer>',
        local_files_only=True,
        model_max_length=512,
        max_len=512,
        truncation=True,
        padding='Longest'
        )
    model1 = AutoModelForSequenceClassification.from_pretrained(
        '<Filepath to fine-tuned model>',
        local_files_only=True,
        problem_type="multi_label_classification", num_labels=18)

    print(get_prediction(discharge_summary, model1, tokenizer))

if __name__ == "__main__":
    visualise_ICD_code_distribution()
    create_subset_of_discharge_summaries()
    create_training_validation_and_test_sets()
    cross_validation()
    fine_tune_model()
    test_model()
    #get_prediction_for_single_discharge_summary("Discharge summaries")

