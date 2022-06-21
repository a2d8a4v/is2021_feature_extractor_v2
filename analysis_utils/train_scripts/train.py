import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import pickleStore, data_processing, countForSegment
# from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer
from model.model import BertClassifier
from model.dataset import Dataset
import os
import math
import argparse
# import multiprocessing as mp


if __name__:
    
    # mp.set_start_method('spawn')

    ## parser initializing
    parser = argparse.ArgumentParser(description='Train classifier model')
    # parser.add_argument('--corr_phn_fn', default="exp/nnet3_chain/model_online/align_lats_kids1500_hires/score_phn_10/kids1500_hires.ctm", type=str)
    parser.add_argument('--state', default="train", type=str)
    parser.add_argument('--ngpu', default=1, type=int, required=False)
    args = parser.parse_args()


    ## Variables
    base = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline"
    feature_type = "normal"
    model = "model_"+feature_type+".pth"
    # dictfile = base+"/data/lang_1char/train_tr_units.txt" # phone-level
    # dictfile = base+"/data/lang_1char/train_tr_units_prompt.txt" # word-level
    dictfile = base+"/data/lang_1char/train_units.txt" # word-level
    dictfile_tgt = {"D": 0, "O": 1} # 0 is also used for padding
    batch_size = 4
    padding = 500
    epoch_n = 1
    checkpoint = 'bert-base-uncased'


    ## Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
    else:
        print("Training by CPU only.")
    # device = torch.device("cpu")


    ## get data
    ## - train data
    train_src, trainidx2utt = data_processing(base+"/data/train", "text_src", device, dictfile, padding, "src", False)
    train_tgt = data_processing(base+"/data/train", "text_tgt", device, dictfile_tgt, padding, "tgt", False)
    trainset = Dataset(train_src, train_tgt)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1) # num_workers should set 1 if put data on CUDA
    print("train data loaded!")
    ## - dev data
    dev_src, devidx2utt = data_processing(base+"/data/dev", "text_src", device, dictfile, padding, "src", False)
    dev_tgt = data_processing(base+"/data/dev", "text_tgt", device, dictfile_tgt, padding, "tgt", False)
    devset = Dataset(dev_src, dev_tgt)
    devloader = torch.utils.data.DataLoader(devset, batch_size=batch_size, shuffle=False, num_workers=1)
    print("dev data loaded!")
    ## - test data
    test_src, testidx2utt = data_processing(base+"/data/dev", "text_src", device, dictfile, padding, "src", False)
    test_tgt = data_processing(base+"/data/dev", "text_tgt", device, dictfile_tgt, padding, "tgt", False)
    testset = Dataset(test_src, test_tgt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    print("test data loaded!")


    ## Create Vocab file for BERT Tokenizer
    ## Retrain a tokenizer
    label_tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True,
    )
    files = ['label-vocab.txt']
    # We use [PAD] only, except '[CLS]', '[SEP]', '[MASK]'
    label_tokenizer.train(
        files,
        vocab_size=2,
        min_frequency=2,
        special_tokens=['[PAD]', '[UNK]'],
        show_progress=False,
        limit_alphabet=1000,
        wordpieces_prefix="##"
    )
    label_tokenizer.save('./bert.vocab.json')
    print("Vocab saved!")

    ## Load from a pre-trained tokenizer
    # @https://stackoverflow.com/questions/62317723/tokens-to-words-mapping-in-the-tokenizer-decode-step-huggingface
    # tokenizer = BertTokenizer.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # test = tokenizer(['@um', 'pizza', 'and', 'pa-', 'and', '@uh', 'pas-', 'pasta'], padding=True, truncation=True, max_length=padding, return_tensors="pt")
    # print(test)
    # input()

    ## generate trainer
    net = BertClassifier(checkpoint=checkpoint, freeze_bert=False, padding=padding)
    net.to(device)



    if not os.path.isfile(base+"/NN/train_scripts/checkpoint/"+model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        for epoch in range(epoch_n):  # loop over the dataset multiple times

            net.train()
            running_loss = 0.0
            train_loss = 0.0
            valid_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, data_index = data
                vocab = inv_map = {v: k for k, v in tokenizer.get_vocab().items()}
                input_enc = tokenizer(list(inputs), padding=True, truncation=True, max_length=padding, return_tensors="pt")
                pro_idx, enc_att = countForSegment(tokenizer, inputs, input_enc, padding) # We do not use enc_att here, we use pro_idx after getting the embedding from net
                label_obj = label_tokenizer.encode_batch(list(labels))
                label_enc = torch.Tensor([x.ids for x in label_obj]).long()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                inputs = input_enc["input_ids"].to(device)
                attention_mask = input_enc["attention_mask"].to(device)
                attention_mask = None
                # print(inputs[0,:])
                # print(attention_mask[0,:])
                # tensors = []
                # for i in inputs[0,:]:
                #     if vocab[i.item()] != '[PAD]':
                #         tensors.append(vocab[i.item()])
                # print(tensors)
                # input()
                labels = label_enc
                outputs = net(inputs, attention_mask, pro_idx)
                loss = criterion(outputs.transpose(1, 2).cpu(), labels.cpu())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            ######################    
            # validate the model #
            ######################
            net.eval()
            # @https://github.com/diegoalejogm/gans/issues/12
            for i, data in enumerate(devloader, 0):
                # move tensors to GPU if CUDA is available
                inputs, labels, data_index = data
                vocab = inv_map = {v: k for k, v in tokenizer.get_vocab().items()}
                input_enc = tokenizer(list(inputs), padding=True, truncation=True, max_length=padding, return_tensors="pt")
                pro_idx, enc_att = countForSegment(tokenizer, inputs, input_enc, padding) # We do not use enc_att here, we use pro_idx after getting the embedding from net
                label_obj = label_tokenizer.encode_batch(list(labels))
                label_enc = torch.Tensor([x.ids for x in label_obj]).long()

                # forward pass: compute predicted outputs by passing inputs to the model
                inputs = input_enc["input_ids"].to(device)
                attention_mask = input_enc["attention_mask"].to(device)
                attention_mask = None
                labels = label_enc

                outputs = net(inputs, attention_mask, pro_idx)
                # calculate the batch loss
                loss = criterion(outputs.transpose(1, 2).cpu(), labels.cpu())
                # update average validation loss 
                valid_loss += loss.item()*inputs.shape[0]
            
            # calculate average losses
            train_loss = train_loss/len(trainloader.dataset)
            valid_loss = valid_loss/len(devloader.dataset)
        
            # print training/validation statistics 
            print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
            torch.save(net.state_dict(), base+"/NN/train_scripts/checkpoint/"+"epoch_"+str(epoch)+"_"+model)
            
        print('Finished Training')

        ## Save model
        torch.save(net.state_dict(), base+"/NN/train_scripts/checkpoint/"+model)
    else:
        net.load_state_dict(torch.load(base+"/NN/train_scripts/checkpoint/"+model))

    ## Save NN classifier decoded data
    # decoded_data = {}
    # with torch.no_grad():
    #     for data in trainloader:
    #         inputs, labels, data_index = data
    #         input_enc = tokenizer(list(inputs), padding=True, truncation=True, max_length=padding, return_tensors="pt")
    #         inputs = input_enc["input_ids"]
    #         attention_mask = input_enc["attention_mask"]
    #         outputs = net(inputs, attention_mask)
    #         for i, d_i in enumerate(data_index.detach().numpy()):
    #             decoded_data[trainidx2utt[d_i]] = outputs.data.detach().numpy()[i,:]
    #     for data in devloader:
    #         inputs, labels, data_index = data
    #         input_enc = tokenizer(list(inputs), padding=True, truncation=True, max_length=padding, return_tensors="pt")
    #         inputs = input_enc["input_ids"]
    #         attention_mask = input_enc["attention_mask"]
    #         outputs = net(inputs, attention_mask)
    #         for i, d_i in enumerate(data_index.detach().numpy()):
    #             decoded_data[devidx2utt[d_i]] = outputs.data.detach().numpy()[i,:]
    #     for data in testloader:
    #         inputs, labels, data_index = data
    #         input_enc = tokenizer(list(inputs), padding=True, truncation=True, max_length=padding, return_tensors="pt")
    #         inputs = input_enc["input_ids"]
    #         attention_mask = input_enc["attention_mask"]
    #         outputs = net(inputs, attention_mask)
    #         for i, d_i in enumerate(data_index.detach().numpy()):
    #             decoded_data[testidx2utt[d_i]] = outputs.data.detach().numpy()[i,:]
    # pickleStore( decoded_data , base+"/NN/train_scripts/decoded_data.pkl" )


    ## - overall accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels, data_index = data
            vocab = inv_map = {v: k for k, v in tokenizer.get_vocab().items()}
            # calculate outputs by running images through the network
            input_enc = tokenizer(list(inputs), padding=True, truncation=True, max_length=padding, return_tensors="pt")
            pro_idx, enc_att = countForSegment(tokenizer, inputs, input_enc, padding) # We do not use enc_att here, we use pro_idx after getting the embedding from net
            inputs = input_enc["input_ids"].to(device)
            label_obj = label_tokenizer.encode_batch(list(labels))
            label_enc = torch.Tensor([x.ids for x in label_obj]).long()
            attention_mask = input_enc["attention_mask"].to(device)
            labels = label_enc.cpu()
            outputs = net(inputs, attention_mask, pro_idx)
            # the class with the highest energy is what we choose as prediction
            max_values, predicted_idx = torch.max(outputs.data.cpu(), dim=2)
            for _i in range(0, len(labels)):
                limit   = (labels[_i,:] != 0).sum().item()
                total  += limit
                correct = (predicted_idx[_i,limit] == labels[_i,limit]).sum().item()
    print('Accuracy of the network: %d %%' % (100 * correct / total))


    ## - each class
    # correct_pred = {classname: 0 for classname in classes}
    # total_pred = {classname: 0 for classname in classes}
    # with torch.no_grad():
    #     for data in testloader:
    #         feats, labels, data_index = data
    #         outputs = net(feats)
    #         _, predictions = torch.max(outputs, 1)
    #         # collect the correct predictions for each class
    #         for label, prediction in zip(labels, predictions):
    #             label = int(label.detach().numpy())
    #             if label == prediction:
    #                 correct_pred[classes[label]] += 1
    #             total_pred[classes[label]] += 1
    # # print accuracy for each class
    # for classname, correct_count in correct_pred.items():
    #     if total_pred[classname] == 0:
    #         accuracy = 0
    #     else:
    #         accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print("Accuracy for class {} is: {:.1f} %".format(classname+1, accuracy))

    ## predict within std
    # print("YANN.predict_within_std")
    # trans_to_expert_std_all = pikleOpen( base+"/NN/trans_to_expert_std_all.pkl" )
    # ## - overall accuracy
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in testloader:
    #         feats, labels, data_index = data
    #         # calculate outputs by running images through the network
    #         outputs = net(feats)
    #         # the class with the highest energy is what we choose as prediction
    #         _, predicted = torch.max(outputs.data, 1)
    #         # print("YANN.predicted",predicted)
    #         al_std = trans_to_expert_std_all["std"]
    #         al_std = torch.tensor([[al_std]*labels.shape[0]]).squeeze(0)
    #         total += labels.size(0)
    #         correct += ((labels - al_std <= predicted)&(predicted <= labels + al_std)).sum().item()
    # print('Accuracy of the network: %d %%' % (100 * correct / total))

    # ## - each class
    # correct_pred = {classname: 0 for classname in classes}
    # total_pred = {classname: 0 for classname in classes}
    # with torch.no_grad():
    #     for data in testloader:
    #         feats, labels, data_index = data
    #         outputs = net(feats)
    #         _, predictions = torch.max(outputs, 1)
    #         # collect the correct predictions for each class
    #         for label, prediction in zip(labels, predictions):
    #             label = int(label.detach().numpy())
    #             if label == prediction:
    #                 correct_pred[classes[label]] += 1
    #             total_pred[classes[label]] += 1
    # # print accuracy for each class
    # for classname, correct_count in correct_pred.items():
    #     if total_pred[classname] == 0:
    #         accuracy = 0
    #     else:
    #         accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print("Accuracy for class {} is: {:.1f} %".format(classname+1, accuracy))