
def import_model(args):
    print("=> creating model {}".format(args.model_name))

    if args.model_name == 'depth_prompt_main':
        from model.ours.depth_prompt_main import depthprompting
        args.prop_kernel = 9
        args.prop_time = 18
        args.conf_prop = True
        args.loss = 'L1L2_SILogloss_init2'
        model = depthprompting(args)

    elif args.model_name == 'COMPLETIONFORMER':
        raise NotImplementedError
    elif args.model_name == 'NLSPN':
        raise NotImplementedError
    else:
        print("Check Model Name ! !")
        raise NotImplementedError
    return model
    