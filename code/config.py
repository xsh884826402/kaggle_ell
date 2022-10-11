# ====================================================
# CFG
# ====================================================
class CFG:
    wandb = False
    competition = 'FB3'
    _wandb_kernel = 'nakama'
    debug = False
    apex = True
    print_freq = 20
    num_workers = 4
    model = "microsoft/deberta-v3-base"
    # model = "../data/model/deberta-v3-base"
    gradient_checkpointing = True
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 6
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 8
    max_len = 512
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 4
    trn_fold = [0, 1, 2, 3]
    train = True

class InferCFG:
    num_workers=4
    path="../data/output/"
    config_path=path+'config.pth'
    model="microsoft/deberta-v3-base"
    gradient_checkpointing=False
    batch_size=24
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    n_fold=4
    trn_fold=[0, 1, 2, 3]
if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]

# ====================================================
# wandb
# ====================================================
if CFG.wandb:
    import wandb

    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        secret_value_0 = user_secrets.get_secret("wandb_api")
        wandb.login(key=secret_value_0)
        anony = None
    except:
        anony = "must"
        print(
            'If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


    run = wandb.init(project='FB3-Public',
                     name=CFG.model,
                     config=class2dict(CFG),
                     group=CFG.model,
                     job_type="train",
                     anonymous=anony)