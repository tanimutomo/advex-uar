import json

import click

from advex_uar.analysis.compute_ata import parse_logs, get_attack,\
    get_attacks, get_defenses, get_defense_from_run_id

def get_defense_run_ids(eval_logs):
    run_ids = []
    for log in eval_logs:
        run_id = log['wandb_run_id']
        if run_id not in run_ids:
            run_ids.append(run_id)
    return run_ids 


def get_attack_types(atas):
    attack_types = {}
    # ata = [[attack_type, epsilon, n_iter, step_size], ATA_value, training_wandb_run_id (if it exists)].
    for ata in atas:
        # attack_type = attack_type
        attack_type = ata[0][0]
        # resemble all ata based on attack_type
        if attack_type not in attack_types:
            attack_types[attack_type] = [ata]
        else:
            attack_types[attack_type].append(ata)
        # attack_types = {attack_type1: [ata1, ata2, ...],
        #                 attack_type2: [ata1, ata2, ...],
        #                 ...}
    return attack_types


def compute_uar(run_id, atas, eval_logs, defense):
    # get the evaluation log of specified run_id
    logs = [log for log in eval_logs if log['wandb_run_id'] == run_id]

    # get the attack method and epsilon when training the defense model
    def_attack = defense[0]
    epsilon = defense[1]

    # attack_types is the dict of atas, whose key is attack_type and value is each ata
    # attack_types = {attack_type1: [ata1, ata2, ...],
    #                 attack_type2: [ata1, ata2, ...],
    #                 ...}
    attack_types = get_attack_types(atas)

    uar_scores = []
    # calcurate UAR for each attack type
    for attack_type, atas in attack_types.items():
        model_acc = 0
        log_count = 0

        # eps used in evaluation
        eps_found = []

        # list of all eps
        eps_vals = [x[0][1] for x in atas]
        # iteration for all evaluation logs
        for log in logs:
            # conditions (1 and 2 and 3)
            # 1. this attack type is same as the current evaluation's one
            # 2. this eval's eps is enough close to any eps in eps_vals to regard them as the same
            # 3. this epsilon's eval log is not counted
            if (log['attack'] == attack_type and
                min(map(lambda x: abs(x - log['epsilon']), eps_vals)) < 0.01 and
                log['epsilon'] not in eps_found):

                # sum all adv acc
                model_acc += log['adv_acc']
                log_count = log_count + 1
                eps_found.append(log['epsilon'])

        # alert that the set of eps is fixed and it is needed for UAR evaluation
        if log_count != 6:
            print('Have {} eval runs for {} ({:9} eps {:.3f}) with attack {:9} instead of 6; saw {}, need {}'
                  .format(log_count, run_id, def_attack, epsilon, attack_type, str(sorted(eps_found)),
                          str(eps_vals)))
        # the set of eps is found in eval_logs
        else:
            # calcurate the UAR ( acc / ata )
            score = model_acc / sum([x[1] for x in atas])
            # append calcurated UAR with the attack_type
            uar_scores.append((attack_type, score))
    return uar_scores

@click.command()
@click.option("--eval_log_file", type=click.Path(exists=True))
@click.option("--calibrated_eps_file", type=click.Path(exists=True))
@click.option("--train_log_file", type=click.Path(exists=True))
@click.option("--out_file", type=click.Path())
@click.option("--run_id", default=None, help="Training run ID to compute UAR for.  If"\
              "not specified, computes for all runs seen.")
@click.option("--max_eps_file", type=click.Path(exists=True))
def main(eval_log_file=None, train_log_file=None, calibrated_eps_file=None, out_file=None,
         run_id=None, max_eps_file=None):
    # load json files as list 
    eval_logs = parse_logs(eval_log_file)
    train_logs = parse_logs(train_log_file)

    # load the json file for eps
    # max_eps is dict ( key: attack method, value: max eps )
    if max_eps_file is not None:
        with open(max_eps_file, 'r') as f:
            max_eps = json.load(f)

    # get all defense parameters for each log in train_logs
    # defenses[i] = [attack, epsilon, n_iters, step_size, run_id, adv_train, std acc]
    # this is the parameters when training defense model
    defenses = get_defenses(train_logs)

    # get all run ids for evaluating uar
    # if run_id is specified in option, evaluate uar for the run_id
    if run_id:
        defense_run_ids = [run_id]
    else:
        defense_run_ids = get_defense_run_ids(eval_logs)
    
    # load calibrate eps file as ata
    # [[attack_type, epsilon, n_iter, step_size], ATA_value, training_wandb_run_id (if it exists)].
    with open(calibrated_eps_file, 'r') as f:
        atas = json.load(f)

    all_scores = []
    # iteration for all run_ids which come from evaluation
    for run_id in defense_run_ids:
        # get all training model parameters of this defense using the current evaluation run_id
        defense = get_defense_from_run_id(run_id, defenses)

        # condition: max_eps_file is defined.
        #            AND
        #            this defense's  method is Not included in max_eps_file.
        if max_eps_file is not None and defense[0] not in max_eps:
            print('Adversarially training against {} not found in max_eps, omitting'.format(defense[0]))
        # condition: max_eps_file is Not defined.
        #            OR
        #            this defense's epsilon is lower than max_eps_file's max.
        elif max_eps_file is None or defense[1] <= max_eps[defense[0]]:
            # compute UAR
            uar_scores = compute_uar(run_id, atas, eval_logs, defense)
            all_scores.append((run_id, defense[0], defense[1], uar_scores))

            # ???
            defense = get_defense_from_run_id(run_id, defenses) # ???

            # report the results
            print('{:8} {:9} Eps: {:.3f}: {}'.format(run_id, defense[0], defense[1], [(x[0], '{:.3f}'.format(x[1])) for x in uar_scores]))

    # write the results in file
    with open(out_file, 'w') as f:
        json.dump(all_scores, f)
        
if __name__ == '__main__':
    main()
