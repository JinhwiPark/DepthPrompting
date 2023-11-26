
def import_model(args):
    print("=> creating model {}".format(args.model_name))

    if args.model_name == 'depth_prompt_main':
        from model.ours.depth_prompt_main import cspn_plusMDE
        args.prop_kernel = 9
        args.prop_time = 18
        args.conf_prop = True
        args.loss = 'L1L2_SILogloss_init2'
        model = cspn_plusMDE(args)

    elif args.model_name == 'COMPLETIONFORMER':
        from model.completionformer.model import CompletionFormer
        model = CompletionFormer(args)

    elif args.model_name == 'NLSPN':
        from model.nlspn.nlspn import nlspn
        model = nlspn(args)

    else:
        print("Check Model Name ! !")
        raise NotImplementedError
    return model
    