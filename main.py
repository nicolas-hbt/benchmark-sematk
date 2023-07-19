from trainer import Trainer
from tester import Tester
from dataset import Dataset
import numpy as np
import pandas as pd
import argparse
import time
import os
import torch
import pickle
import json
from datetime import datetime

date_today = datetime.today().strftime('%d-%m-%Y')

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', default=400, type=int, help="number of epochs")
    parser.add_argument('-lr', default=0.05, type=float, help="learning rate")
    parser.add_argument('-reg', default=0.00001, type=float, help="l2 regularization parameter")
    parser.add_argument('-margin', default=2.0, type=float, help="margin")
    parser.add_argument('-dataset', default="Codex-S", type=str, help="dataset")
    parser.add_argument('-model', default="TransE", type=str, help="knowledge graph embedding model")
    parser.add_argument('-dim', default=100, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=1, type=int, help="number of negative examples per positive example")
    parser.add_argument('-neg_sampler', default="rns", type=str, help="negative sampling strategy")
    parser.add_argument('-batch_size', default=512, type=int, help="batch size")
    parser.add_argument('-save_each', default=10, type=int, help="validate every k epochs")
    parser.add_argument('-criterion_validation', default="mrr", type=str, help="criterion for keeping best epoch")
    parser.add_argument('-metrics', default="all", type=str, help="metrics to compute on test set (sem|ranks|all)")
    parser.add_argument('-pipeline', default="both", type=str, help="(train|test|both)")
    parser.add_argument('-loss', default="margin-based", type=str, help="(margin-based|logistic|adversarial)")
    parser.add_argument('-device', default="cuda:0", type=str, help="(cpu|cuda:0)")
    parser.add_argument('-setting', default="both", type=str, help="CWA|OWA|both")
    parser.add_argument('-sem', default="both", type=str, help="schema|extensional|both")
    parser.add_argument('-hierarchy', default=True, type=bool)
    parser.add_argument('-resume_training', default=False, type=bool)
    parser.add_argument('-resume_epoch', default=0, type=int, help='epoch at which resuming training (0 means: last epoch the model was saved)')
    parser.add_argument('-test_one_epoch', default=False, type=bool)

    #for ConvE
    parser.add_argument('-input_drop', default=0.2, type=float)
    parser.add_argument('-hidden_drop', default=0.3, type=float)
    parser.add_argument('-feat_drop', default=0.3, type=float)
    parser.add_argument('-hidden_size', default=9728, type=int)
    parser.add_argument('-embedding_shape1', default=20, type=int)

    #for ConvKB
    parser.add_argument('-hid_convkb', default=20, type=int)
    parser.add_argument('-num_of_filters_convkb', default=16, type=int)

    #for R-GCN
    parser.add_argument("-graph-batch-size", type=int, default=30000)
    parser.add_argument("-graph-split-size", type=float, default=0.5)
    
    parser.add_argument("-dropout", type=float, default=0.2)
    parser.add_argument("-gpu", type=int, default=-1)
    parser.add_argument("-num_bases", type=int, default=4)
    
    parser.add_argument("-grad-norm", type=float, default=1.0)

    args = parser.parse_args()
    np.random.seed(7)
    torch.manual_seed(7)
    return args

if __name__ == '__main__':
    args = get_parameter()
    dataset = Dataset(args.dataset, args)
    model = args.model

    if args.pipeline == 'both' or args.pipeline == 'train':
        if args.resume_training == False:
            print("------- Training -------")
            start = time.time()
            trainer = Trainer(dataset, model, args)
            trainer.train()
        else:
            print("------- Training -------")
            start = time.time()
            trainer = Trainer(dataset, model, args)
            trainer.resume_training()
        print("Training time: ", time.time() - start)

    if args.pipeline == 'both' or args.pipeline == 'test':
        print("------- Select best epoch on validation set -------")
        if args.resume_training == False:
            epochs2test = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
        else:
            if args.resume_epoch == 0:
                # get last pt file
                resume_epoch = max([int(f[-11:].split('=')[-1].split('.')[0]) for f in os.listdir("models/" + str(dataset.name) + "/" + str(model) + "/")]) - args.ne
            else:
                resume_epoch = args.resume_epoch
            print('Resuming at epoch ' + str(resume_epoch))
            epochs2test = [str(int(resume_epoch + args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
        if args.test_one_epoch == True:
            resume_epoch = args.resume_epoch
            print(resume_epoch)
            epochs2test = [str(resume_epoch)]

        dataset = Dataset(args.dataset, args)
        
        best_mrr = -1.0
        best_loss = + np.inf
        results = {}
        best_epoch = "0"
        directory = "models/" + dataset.name + "/" + model + "/"
        if not os.path.exists('results/' + dataset.name + "/" + args.model + '/' + args.sem+'-'+args.setting + '/'):
            os.makedirs('results/' + dataset.name + "/" + args.model + '/' + args.sem+'-'+args.setting + '/')
        for epoch in epochs2test:
            print("Epoch n°", epoch)
            model_path = directory + "dim="+str(args.dim) + \
        "_lr="+str(args.lr) + "_neg="+str(args.neg_ratio) + "_bs="+str(args.batch_size) + "_reg="+str(args.reg)+ "__epoch="+str(epoch) + ".pt"
            tester = Tester(dataset, args, model_path, "valid")
            start = time.time()
            if args.criterion_validation == 'mrr':
                if args.metrics == 'sem' or args.metrics == 'all':
                    filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10, filt_mrr_h, filt_mrr_t, \
                    filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, schema_OWA, ext_CWA, schema_WUP = tester.calc_valid_mrr()

                    if args.sem == 'both':
                        if args.setting == 'both':
                            results[int(epoch)] = {
                            "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                            "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                            "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                            "CWA_Sem@1": schema_CWA['sem1'], "CWA_Sem@1_h": schema_CWA['sem1_h'], "CWA_Sem@1_t": schema_CWA['sem1_t'], "CWA_Sem@3": schema_CWA['sem3'], \
                            "CWA_Sem@3_h": schema_CWA['sem3_h'], "CWA_Sem@3_t": schema_CWA['sem3_t'], "CWA_Sem@5": schema_CWA['sem5'], "CWA_Sem@5_h": schema_CWA['sem5_h'], \
                            "CWA_Sem@5_t": schema_CWA['sem5_t'], "CWA_Sem@10": schema_CWA['sem10'], "CWA_Sem@10_h": schema_CWA['sem10_h'], "CWA_Sem@10_t": schema_CWA['sem10_t'], \
                            
                            "OWA_Sem@1": schema_OWA['sem1'], "OWA_Sem@1_h": schema_OWA['sem1_h'], "OWA_Sem@1_t": schema_OWA['sem1_t'], "OWA_Sem@3": schema_OWA['sem3'], \
                            "OWA_Sem@3_h": schema_OWA['sem3_h'], "OWA_Sem@3_t": schema_OWA['sem3_t'], "OWA_Sem@5": schema_OWA['sem5'], "OWA_Sem@5_h": schema_OWA['sem5_h'], \
                            "OWA_Sem@5_t": schema_OWA['sem5_t'], "OWA_Sem@10": schema_OWA['sem10'], "OWA_Sem@10_h": schema_OWA['sem10_h'], "OWA_Sem@10_t": schema_OWA['sem10_t'], \
                            
                            "Ext_Sem@1": ext_CWA['sem1'], "Ext_Sem@1_h": ext_CWA['sem1_h'], "Ext_Sem@1_t": ext_CWA['sem1_t'], "Ext_Sem@3": ext_CWA['sem3'], \
                            "Ext_Sem@3_h": ext_CWA['sem3_h'], "Ext_Sem@3_t": ext_CWA['sem3_t'], "Ext_Sem@5": ext_CWA['sem5'], "Ext_Sem@5_h": ext_CWA['sem5_h'], \
                            "Ext_Sem@5_t": ext_CWA['sem5_t'], "Ext_Sem@10": ext_CWA['sem10'], "Ext_Sem@10_h": ext_CWA['sem10_h'], "Ext_Sem@10_t": ext_CWA['sem10_t']}
                        elif args.setting == 'CWA':
                            results[int(epoch)] = {
                            "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                            "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                            "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                            "CWA_Sem@1": schema_CWA['sem1'], "CWA_Sem@1_h": schema_CWA['sem1_h'], "CWA_Sem@1_t": schema_CWA['sem1_t'], "CWA_Sem@3": schema_CWA['sem3'], \
                            "CWA_Sem@3_h": schema_CWA['sem3_h'], "CWA_Sem@3_t": schema_CWA['sem3_t'], "CWA_Sem@5": schema_CWA['sem5'], "CWA_Sem@5_h": schema_CWA['sem5_h'], \
                            "CWA_Sem@5_t": schema_CWA['sem5_t'], "CWA_Sem@10": schema_CWA['sem10'], "CWA_Sem@10_h": schema_CWA['sem10_h'], "CWA_Sem@10_t": schema_CWA['sem10_t'], \

                            "WUP_Sem@1": schema_WUP['sem1'], "WUP_Sem@1_h": schema_WUP['sem1_h'], "WUP_Sem@1_t": schema_WUP['sem1_t'], "WUP_Sem@3": schema_WUP['sem3'], \
                            "WUP_Sem@3_h": schema_WUP['sem3_h'], "WUP_Sem@3_t": schema_WUP['sem3_t'], "WUP_Sem@5": schema_WUP['sem5'], "WUP_Sem@5_h": schema_WUP['sem5_h'], \
                            "WUP_Sem@5_t": schema_WUP['sem5_t'], "WUP_Sem@10": schema_WUP['sem10'], "WUP_Sem@10_h": schema_WUP['sem10_h'], "WUP_Sem@10_t": schema_WUP['sem10_t'], \
                            
                            "Ext_Sem@1": ext_CWA['sem1'], "Ext_Sem@1_h": ext_CWA['sem1_h'], "Ext_Sem@1_t": ext_CWA['sem1_t'], "Ext_Sem@3": ext_CWA['sem3'], \
                            "Ext_Sem@3_h": ext_CWA['sem3_h'], "Ext_Sem@3_t": ext_CWA['sem3_t'], "Ext_Sem@5": ext_CWA['sem5'], "Ext_Sem@5_h": ext_CWA['sem5_h'], \
                            "Ext_Sem@5_t": ext_CWA['sem5_t'], "Ext_Sem@10": ext_CWA['sem10'], "Ext_Sem@10_h": ext_CWA['sem10_h'], "Ext_Sem@10_t": ext_CWA['sem10_t']}
                        else:
                            results[int(epoch)] = {
                            "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                            "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                            "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                            "OWA_Sem@1": schema_OWA['sem1'], "OWA_Sem@1_h": schema_OWA['sem1_h'], "OWA_Sem@1_t": schema_OWA['sem1_t'], "OWA_Sem@3": schema_OWA['sem3'], \
                            "OWA_Sem@3_h": schema_OWA['sem3_h'], "OWA_Sem@3_t": schema_OWA['sem3_t'], "OWA_Sem@5": schema_OWA['sem5'], "OWA_Sem@5_h": schema_OWA['sem5_h'], \
                            "OWA_Sem@5_t": schema_OWA['sem5_t'], "OWA_Sem@10": schema_OWA['sem10'], "OWA_Sem@10_h": schema_OWA['sem10_h'], "OWA_Sem@10_t": schema_OWA['sem10_t'], \
                            
                            "Ext_Sem@1": ext_CWA['sem1'], "Ext_Sem@1_h": ext_CWA['sem1_h'], "Ext_Sem@1_t": ext_CWA['sem1_t'], "Ext_Sem@3": ext_CWA['sem3'], \
                            "Ext_Sem@3_h": ext_CWA['sem3_h'], "Ext_Sem@3_t": ext_CWA['sem3_t'], "Ext_Sem@5": ext_CWA['sem5'], "Ext_Sem@5_h": ext_CWA['sem5_h'], \
                            "Ext_Sem@5_t": ext_CWA['sem5_t'], "Ext_Sem@10": ext_CWA['sem10'], "Ext_Sem@10_h": ext_CWA['sem10_h'], "Ext_Sem@10_t": ext_CWA['sem10_t']}
                    elif args.sem == 'schema':
                        if args.setting == 'both':
                            results[int(epoch)] = {
                            "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                            "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                            "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                            "CWA_Sem@1": schema_CWA['sem1'], "CWA_Sem@1_h": schema_CWA['sem1_h'], "CWA_Sem@1_t": schema_CWA['sem1_t'], "CWA_Sem@3": schema_CWA['sem3'], \
                            "CWA_Sem@3_h": schema_CWA['sem3_h'], "CWA_Sem@3_t": schema_CWA['sem3_t'], "CWA_Sem@5": schema_CWA['sem5'], "CWA_Sem@5_h": schema_CWA['sem5_h'], \
                            "CWA_Sem@5_t": schema_CWA['sem5_t'], "CWA_Sem@10": schema_CWA['sem10'], "CWA_Sem@10_h": schema_CWA['sem10_h'], "CWA_Sem@10_t": schema_CWA['sem10_t'], \
                            
                            "OWA_Sem@1": schema_OWA['sem1'], "OWA_Sem@1_h": schema_OWA['sem1_h'], "OWA_Sem@1_t": schema_OWA['sem1_t'], "OWA_Sem@3": schema_OWA['sem3'], \
                            "OWA_Sem@3_h": schema_OWA['sem3_h'], "OWA_Sem@3_t": schema_OWA['sem3_t'], "OWA_Sem@5": schema_OWA['sem5'], "OWA_Sem@5_h": schema_OWA['sem5_h'], \
                            "OWA_Sem@5_t": schema_OWA['sem5_t'], "OWA_Sem@10": schema_OWA['sem10'], "OWA_Sem@10_h": schema_OWA['sem10_h'], "OWA_Sem@10_t": schema_OWA['sem10_t']}
                        elif args.setting == 'CWA':
                            results[int(epoch)] = {
                            "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                            "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                            "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                            "WUP_Sem@1": schema_WUP['sem1'], "WUP_Sem@1_h": schema_WUP['sem1_h'], "WUP_Sem@1_t": schema_WUP['sem1_t'], "WUP_Sem@3": schema_WUP['sem3'], \
                            "WUP_Sem@3_h": schema_WUP['sem3_h'], "WUP_Sem@3_t": schema_WUP['sem3_t'], "WUP_Sem@5": schema_WUP['sem5'], "WUP_Sem@5_h": schema_WUP['sem5_h'], \
                            "WUP_Sem@5_t": schema_WUP['sem5_t'], "WUP_Sem@10": schema_WUP['sem10'], "WUP_Sem@10_h": schema_WUP['sem10_h'], "WUP_Sem@10_t": schema_WUP['sem10_t'], \

                            "CWA_Sem@1": schema_CWA['sem1'], "CWA_Sem@1_h": schema_CWA['sem1_h'], "CWA_Sem@1_t": schema_CWA['sem1_t'], "CWA_Sem@3": schema_CWA['sem3'], \
                            "CWA_Sem@3_h": schema_CWA['sem3_h'], "CWA_Sem@3_t": schema_CWA['sem3_t'], "CWA_Sem@5": schema_CWA['sem5'], "CWA_Sem@5_h": schema_CWA['sem5_h'], \
                            "CWA_Sem@5_t": schema_CWA['sem5_t'], "CWA_Sem@10": schema_CWA['sem10'], "CWA_Sem@10_h": schema_CWA['sem10_h'], "CWA_Sem@10_t": schema_CWA['sem10_t']}
                        else:
                            results[int(epoch)] = {
                            "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                            "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                            "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                            "OWA_Sem@1": schema_OWA['sem1'], "OWA_Sem@1_h": schema_OWA['sem1_h'], "OWA_Sem@1_t": schema_OWA['sem1_t'], "OWA_Sem@3": schema_OWA['sem3'], \
                            "OWA_Sem@3_h": schema_OWA['sem3_h'], "OWA_Sem@3_t": schema_OWA['sem3_t'], "OWA_Sem@5": schema_OWA['sem5'], "OWA_Sem@5_h": schema_OWA['sem5_h'], \
                            "OWA_Sem@5_t": schema_OWA['sem5_t'], "OWA_Sem@10": schema_OWA['sem10'], "OWA_Sem@10_h": schema_OWA['sem10_h'], "OWA_Sem@10_t": schema_OWA['sem10_t']}
                    elif args.sem == 'extensional':
                            results[int(epoch)] = {
                            "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                            "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                            "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                            "Ext_Sem@1": ext_CWA['sem1'], "Ext_Sem@1_h": ext_CWA['sem1_h'], "Ext_Sem@1_t": ext_CWA['sem1_t'], "Ext_Sem@3": ext_CWA['sem3'], \
                            "Ext_Sem@3_h": ext_CWA['sem3_h'], "Ext_Sem@3_t": ext_CWA['sem3_t'], "Ext_Sem@5": ext_CWA['sem5'], "Ext_Sem@5_h": ext_CWA['sem5_h'], \
                            "Ext_Sem@5_t": ext_CWA['sem5_t'], "Ext_Sem@10": ext_CWA['sem10'], "Ext_Sem@10_h": ext_CWA['sem10_h'], "Ext_Sem@10_t": ext_CWA['sem10_t']}

                    df_valid = pd.DataFrame.from_dict(results) 
                    df_valid.to_csv ('results/' + dataset.name + "/" + args.model + '/' + args.sem+'-'+args.setting + '/' + '['+ date_today + ']'+ args.model + '-valid_results-mrr-sem-' + args.dataset + '-' + model + '-dim=' + \
                str(args.dim) + '-lr=' + str(args.lr) + "-reg="+str(args.reg) + '-neg=' + str(args.neg_ratio) + '-bs=' + str(args.batch_size) + '.csv')
                    # JSON
                    with open('results/' + dataset.name + "/" + args.model + '/' + args.sem+'-'+args.setting + '/' '['+ date_today + ']'+ args.model + '-valid_results-mrr-sem-' + args.dataset + '-' + model + '-dim=' + \
                str(args.dim) + '-lr=' + str(args.lr) + "-reg="+str(args.reg) + '-neg=' + str(args.neg_ratio) + '-bs=' + str(args.batch_size) + '.json', 'w') as fp:
                        json.dump(results, fp)
                else:
                    filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10,\
                    filt_mrr_h, filt_mrr_t, filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t = tester.calc_valid_mrr()

                    results[int(epoch)] = {"MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                    "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                    "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t}

                    df_valid = pd.DataFrame.from_dict(results) 
                    df_valid.to_csv ('results/' + dataset.name + "/" + args.model + '/' + args.sem+'-'+args.setting + '/' + '['+ date_today + ']'+ args.model + '-valid_results-mrr_only-' + args.dataset + '-' + model + '-dim=' + \
                str(args.dim) + '-lr=' + str(args.lr) + "-reg="+str(args.reg) + '-neg=' + str(args.neg_ratio) + '-bs=' + str(args.batch_size) + '.csv')
                    # JSON
                    with open('results/' + dataset.name + "/" + args.model + '/' + args.sem+'-'+args.setting + '/' + '['+ date_today + ']'+ args.model + '-valid_results-mrr_only-' + args.dataset + '-' + model + '-dim=' + \
                str(args.dim) + '-lr=' + str(args.lr) + "-reg="+str(args.reg) + '-neg=' + str(args.neg_ratio) + '-bs=' + str(args.batch_size) +'.json', 'w') as fp:
                        json.dump(results, fp)

                if filt_mrr > best_mrr:
                    best_mrr = filt_mrr
                    best_epoch = epoch
            print(time.time() - start)
        print("Best epoch: " + best_epoch)

        print("------- Testing on the best epoch -------")
        best_model_path = directory + "dim="+str(args.dim) + \
        "_lr="+str(args.lr) + "_neg="+str(args.neg_ratio) + "_bs="+str(args.batch_size) + "_reg="+str(args.reg)+ "__epoch="+ best_epoch + ".pt"
        tester = Tester(dataset, args, best_model_path, "test")
        if args.metrics == 'sem' or args.metrics == 'all':
            filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10,\
            filt_mrr_h, filt_mrr_t, filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, schema_OWA, ext_CWA, schema_WUP = tester.test()

            if args.sem == 'both':
                if args.setting == 'both':
                    best_ep_results = {
                    "Epoch": int(best_epoch), "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                    "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                    "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                    "CWA_Sem@1": schema_CWA['sem1'], "CWA_Sem@1_h": schema_CWA['sem1_h'], "CWA_Sem@1_t": schema_CWA['sem1_t'], "CWA_Sem@3": schema_CWA['sem3'], \
                    "CWA_Sem@3_h": schema_CWA['sem3_h'], "CWA_Sem@3_t": schema_CWA['sem3_t'], "CWA_Sem@5": schema_CWA['sem5'], "CWA_Sem@5_h": schema_CWA['sem5_h'], \
                    "CWA_Sem@5_t": schema_CWA['sem5_t'], "CWA_Sem@10": schema_CWA['sem10'], "CWA_Sem@10_h": schema_CWA['sem10_h'], "CWA_Sem@10_t": schema_CWA['sem10_t'], \
                    
                    "OWA_Sem@1": schema_OWA['sem1'], "OWA_Sem@1_h": schema_OWA['sem1_h'], "OWA_Sem@1_t": schema_OWA['sem1_t'], "OWA_Sem@3": schema_OWA['sem3'], \
                    "OWA_Sem@3_h": schema_OWA['sem3_h'], "OWA_Sem@3_t": schema_OWA['sem3_t'], "OWA_Sem@5": schema_OWA['sem5'], "OWA_Sem@5_h": schema_OWA['sem5_h'], \
                    "OWA_Sem@5_t": schema_OWA['sem5_t'], "OWA_Sem@10": schema_OWA['sem10'], "OWA_Sem@10_h": schema_OWA['sem10_h'], "OWA_Sem@10_t": schema_OWA['sem10_t'], \
                    
                    "Ext_Sem@1": ext_CWA['sem1'], "Ext_Sem@1_h": ext_CWA['sem1_h'], "Ext_Sem@1_t": ext_CWA['sem1_t'], "Ext_Sem@3": ext_CWA['sem3'], \
                    "Ext_Sem@3_h": ext_CWA['sem3_h'], "Ext_Sem@3_t": ext_CWA['sem3_t'], "Ext_Sem@5": ext_CWA['sem5'], "Ext_Sem@5_h": ext_CWA['sem5_h'], \
                    "Ext_Sem@5_t": ext_CWA['sem5_t'], "Ext_Sem@10": ext_CWA['sem10'], "Ext_Sem@10_h": ext_CWA['sem10_h'], "Ext_Sem@10_t": ext_CWA['sem10_t']}
                elif args.setting == 'CWA':
                    best_ep_results = {
                    "Epoch": int(best_epoch), "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                    "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                    "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                    "CWA_Sem@1": schema_CWA['sem1'], "CWA_Sem@1_h": schema_CWA['sem1_h'], "CWA_Sem@1_t": schema_CWA['sem1_t'], "CWA_Sem@3": schema_CWA['sem3'], \
                    "CWA_Sem@3_h": schema_CWA['sem3_h'], "CWA_Sem@3_t": schema_CWA['sem3_t'], "CWA_Sem@5": schema_CWA['sem5'], "CWA_Sem@5_h": schema_CWA['sem5_h'], \
                    "CWA_Sem@5_t": schema_CWA['sem5_t'], "CWA_Sem@10": schema_CWA['sem10'], "CWA_Sem@10_h": schema_CWA['sem10_h'], "CWA_Sem@10_t": schema_CWA['sem10_t'], \

                    "WUP_Sem@1": schema_WUP['sem1'], "WUP_Sem@1_h": schema_WUP['sem1_h'], "WUP_Sem@1_t": schema_WUP['sem1_t'], "WUP_Sem@3": schema_WUP['sem3'], \
                    "WUP_Sem@3_h": schema_WUP['sem3_h'], "WUP_Sem@3_t": schema_WUP['sem3_t'], "WUP_Sem@5": schema_WUP['sem5'], "WUP_Sem@5_h": schema_WUP['sem5_h'], \
                    "WUP_Sem@5_t": schema_WUP['sem5_t'], "WUP_Sem@10": schema_WUP['sem10'], "WUP_Sem@10_h": schema_WUP['sem10_h'], "WUP_Sem@10_t": schema_WUP['sem10_t'], \
                    
                    "Ext_Sem@1": ext_CWA['sem1'], "Ext_Sem@1_h": ext_CWA['sem1_h'], "Ext_Sem@1_t": ext_CWA['sem1_t'], "Ext_Sem@3": ext_CWA['sem3'], \
                    "Ext_Sem@3_h": ext_CWA['sem3_h'], "Ext_Sem@3_t": ext_CWA['sem3_t'], "Ext_Sem@5": ext_CWA['sem5'], "Ext_Sem@5_h": ext_CWA['sem5_h'], \
                    "Ext_Sem@5_t": ext_CWA['sem5_t'], "Ext_Sem@10": ext_CWA['sem10'], "Ext_Sem@10_h": ext_CWA['sem10_h'], "Ext_Sem@10_t": ext_CWA['sem10_t']}
                else:
                    best_ep_results = {
                    "Epoch": int(best_epoch), "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                    "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                    "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                    "OWA_Sem@1": schema_OWA['sem1'], "OWA_Sem@1_h": schema_OWA['sem1_h'], "OWA_Sem@1_t": schema_OWA['sem1_t'], "OWA_Sem@3": schema_OWA['sem3'], \
                    "OWA_Sem@3_h": schema_OWA['sem3_h'], "OWA_Sem@3_t": schema_OWA['sem3_t'], "OWA_Sem@5": schema_OWA['sem5'], "OWA_Sem@5_h": schema_OWA['sem5_h'], \
                    "OWA_Sem@5_t": schema_OWA['sem5_t'], "OWA_Sem@10": schema_OWA['sem10'], "OWA_Sem@10_h": schema_OWA['sem10_h'], "OWA_Sem@10_t": schema_OWA['sem10_t'], \
                    
                    "Ext_Sem@1": ext_CWA['sem1'], "Ext_Sem@1_h": ext_CWA['sem1_h'], "Ext_Sem@1_t": ext_CWA['sem1_t'], "Ext_Sem@3": ext_CWA['sem3'], \
                    "Ext_Sem@3_h": ext_CWA['sem3_h'], "Ext_Sem@3_t": ext_CWA['sem3_t'], "Ext_Sem@5": ext_CWA['sem5'], "Ext_Sem@5_h": ext_CWA['sem5_h'], \
                    "Ext_Sem@5_t": ext_CWA['sem5_t'], "Ext_Sem@10": ext_CWA['sem10'], "Ext_Sem@10_h": ext_CWA['sem10_h'], "Ext_Sem@10_t": ext_CWA['sem10_t']}
            elif args.sem == 'schema':
                if args.setting == 'both':
                    best_ep_results = {
                    "Epoch": int(best_epoch), "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                    "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                    "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                    "CWA_Sem@1": schema_CWA['sem1'], "CWA_Sem@1_h": schema_CWA['sem1_h'], "CWA_Sem@1_t": schema_CWA['sem1_t'], "CWA_Sem@3": schema_CWA['sem3'], \
                    "CWA_Sem@3_h": schema_CWA['sem3_h'], "CWA_Sem@3_t": schema_CWA['sem3_t'], "CWA_Sem@5": schema_CWA['sem5'], "CWA_Sem@5_h": schema_CWA['sem5_h'], \
                    "CWA_Sem@5_t": schema_CWA['sem5_t'], "CWA_Sem@10": schema_CWA['sem10'], "CWA_Sem@10_h": schema_CWA['sem10_h'], "CWA_Sem@10_t": schema_CWA['sem10_t'], \

                    "WUP_Sem@1": schema_WUP['sem1'], "WUP_Sem@1_h": schema_WUP['sem1_h'], "WUP_Sem@1_t": schema_WUP['sem1_t'], "WUP_Sem@3": schema_WUP['sem3'], \
                    "WUP_Sem@3_h": schema_WUP['sem3_h'], "WUP_Sem@3_t": schema_WUP['sem3_t'], "WUP_Sem@5": schema_WUP['sem5'], "WUP_Sem@5_h": schema_WUP['sem5_h'], \
                    "WUP_Sem@5_t": schema_WUP['sem5_t'], "WUP_Sem@10": schema_WUP['sem10'], "WUP_Sem@10_h": schema_WUP['sem10_h'], "WUP_Sem@10_t": schema_WUP['sem10_t'], \
                    
                    "OWA_Sem@1": schema_OWA['sem1'], "OWA_Sem@1_h": schema_OWA['sem1_h'], "OWA_Sem@1_t": schema_OWA['sem1_t'], "OWA_Sem@3": schema_OWA['sem3'], \
                    "OWA_Sem@3_h": schema_OWA['sem3_h'], "OWA_Sem@3_t": schema_OWA['sem3_t'], "OWA_Sem@5": schema_OWA['sem5'], "OWA_Sem@5_h": schema_OWA['sem5_h'], \
                    "OWA_Sem@5_t": schema_OWA['sem5_t'], "OWA_Sem@10": schema_OWA['sem10'], "OWA_Sem@10_h": schema_OWA['sem10_h'], "OWA_Sem@10_t": schema_OWA['sem10_t']}
                elif args.setting == 'CWA':
                    best_ep_results = {
                    "Epoch": int(best_epoch), "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                    "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                    "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                    "WUP_Sem@1": schema_WUP['sem1'], "WUP_Sem@1_h": schema_WUP['sem1_h'], "WUP_Sem@1_t": schema_WUP['sem1_t'], "WUP_Sem@3": schema_WUP['sem3'], \
                    "WUP_Sem@3_h": schema_WUP['sem3_h'], "WUP_Sem@3_t": schema_WUP['sem3_t'], "WUP_Sem@5": schema_WUP['sem5'], "WUP_Sem@5_h": schema_WUP['sem5_h'], \
                    "WUP_Sem@5_t": schema_WUP['sem5_t'], "WUP_Sem@10": schema_WUP['sem10'], "WUP_Sem@10_h": schema_WUP['sem10_h'], "WUP_Sem@10_t": schema_WUP['sem10_t'], \

                    "CWA_Sem@1": schema_CWA['sem1'], "CWA_Sem@1_h": schema_CWA['sem1_h'], "CWA_Sem@1_t": schema_CWA['sem1_t'], "CWA_Sem@3": schema_CWA['sem3'], \
                    "CWA_Sem@3_h": schema_CWA['sem3_h'], "CWA_Sem@3_t": schema_CWA['sem3_t'], "CWA_Sem@5": schema_CWA['sem5'], "CWA_Sem@5_h": schema_CWA['sem5_h'], \
                    "CWA_Sem@5_t": schema_CWA['sem5_t'], "CWA_Sem@10": schema_CWA['sem10'], "CWA_Sem@10_h": schema_CWA['sem10_h'], "CWA_Sem@10_t": schema_CWA['sem10_t']}
                else:
                    best_ep_results = {
                    "Epoch": int(best_epoch), "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                    "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                    "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                    "OWA_Sem@1": schema_OWA['sem1'], "OWA_Sem@1_h": schema_OWA['sem1_h'], "OWA_Sem@1_t": schema_OWA['sem1_t'], "OWA_Sem@3": schema_OWA['sem3'], \
                    "OWA_Sem@3_h": schema_OWA['sem3_h'], "OWA_Sem@3_t": schema_OWA['sem3_t'], "OWA_Sem@5": schema_OWA['sem5'], "OWA_Sem@5_h": schema_OWA['sem5_h'], \
                    "OWA_Sem@5_t": schema_OWA['sem5_t'], "OWA_Sem@10": schema_OWA['sem10'], "OWA_Sem@10_h": schema_OWA['sem10_h'], "OWA_Sem@10_t": schema_OWA['sem10_t']}
            elif args.sem == 'extensional':
                    best_ep_results = {
                    "Epoch": int(best_epoch), "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                    "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                    "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t, \

                    "Ext_Sem@1": ext_CWA['sem1'], "Ext_Sem@1_h": ext_CWA['sem1_h'], "Ext_Sem@1_t": ext_CWA['sem1_t'], "Ext_Sem@3": ext_CWA['sem3'], \
                    "Ext_Sem@3_h": ext_CWA['sem3_h'], "Ext_Sem@3_t": ext_CWA['sem3_t'], "Ext_Sem@5": ext_CWA['sem5'], "Ext_Sem@5_h": ext_CWA['sem5_h'], \
                    "Ext_Sem@5_t": ext_CWA['sem5_t'], "Ext_Sem@10": ext_CWA['sem10'], "Ext_Sem@10_h": ext_CWA['sem10_h'], "Ext_Sem@10_t": ext_CWA['sem10_t']}

            df_test = pd.DataFrame.from_dict([best_ep_results]) 
            df_test.to_csv ('results/' + dataset.name + "/" + args.model + '/' + args.sem+'-'+args.setting + '/' + '['+ date_today + ']best-epoch_results-ALL-METRICS-' + args.dataset + '-' + model + '-dim=' + \
                str(args.dim) + '-lr=' + str(args.lr) + "-reg="+str(args.reg) + '-neg=' + str(args.neg_ratio) + '-bs=' + str(args.batch_size) + '.csv')

            with open('results/' + dataset.name + "/" + args.model + '/' + args.sem+'-'+args.setting + '/' + '['+ date_today + ']best-epoch_results-ALL-METRICS-' + args.dataset + '-' + model + '-dim=' + \
                str(args.dim) + '-lr=' + str(args.lr) + "-reg="+str(args.reg) + '-neg=' + str(args.neg_ratio) + '-bs=' + str(args.batch_size) + '.json', 'w') as fp:
                json.dump(best_ep_results, fp)
        else:

            filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10,\
            filt_mrr_h, filt_mrr_t, filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t = tester.test()

            best_ep_results = {"Epoch": best_epoch, "MRR": filt_mrr, "MRR_h": filt_mrr_h, "MRR_t": filt_mrr_t, "H@1": filtered_hits_at_1, "H@1_h": filt_h1_h, "H@1_t": filt_h1_t,\
                    "H@3": filtered_hits_at_3, "H@3_h": filt_h3_h, "H@3_t": filt_h3_t, "H@5": filtered_hits_at_5, "H@5_h": filt_h5_h, "H@5_t": filt_h5_t, \
                    "H@10": filtered_hits_at_10, "H@10_h": filt_h10_h, "H@10_t": filt_h10_t}

            df_test = pd.DataFrame.from_dict([best_ep_results]) 
            df_test.to_csv ('results/' + dataset.name + "/" + args.model + '/' + args.sem+'-'+args.setting + '/' + '['+ date_today + ']best-epoch_results-RANK-BASED-' + args.dataset + '-' + model + '-dim=' + \
                str(args.dim) + '-lr=' + str(args.lr) + "-reg="+str(args.reg) + '-neg=' + str(args.neg_ratio) + '-bs=' + str(args.batch_size) + '.csv')

            with open('results/' + dataset.name + "/" + args.model + '/' + args.sem+'-'+args.setting + '/' + '['+ date_today + ']best-epoch_results-RANK-BASED-' + args.dataset + '-' + model + '-dim=' + \
                str(args.dim) + '-lr=' + str(args.lr) + "-reg="+str(args.reg) + '-neg=' + str(args.neg_ratio) + '-bs=' + str(args.batch_size) +'.json', 'w') as fp:
                json.dump(best_ep_results, fp)