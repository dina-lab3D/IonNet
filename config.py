base_config = dict(
    model_name="GAT_CONV",
    split_mode="DEFAULT",
    gat_dict=dict(
        c_in=16,
        c_hidden=16,
        c_out=1,
        head1=12,
        head2=11,
        head3=4,
        dp_rate=0,
        edge_dim=1
    ),
    model_dict=dict(
        c_in=16,
        c_out=1,
        c_hidden=16,
        num_layers=3,
        dp_rate=0,
        heads=8,
        concat=False
    ),
    conv_dict=dict(
        c_in=16,
        c_out=1,
        c_hidden=16,
        num_layers=9,
        lstm_steps=17
    ),
    train_dict=dict(
        batch_size=200,
        lr=0.00009591254716139546,
        weight_decay=1e-1,
        epochs=200,
        num_workers=2,
        num_of_classes=1,
        pos_weight=1.8828560255842843,
    ),
    gat_conv_dict=dict(
      c_in=16,
      c_hidden=32,
      c_out=1,
      heads=10,
      dp_rate=0.2,
      edge_dim=1,
      num_layers=3,
      lstm_steps=12
    ),
    test_dict=dict(
        thresh=[0.72, 0.75, 1.0, 1.5, 2.0, 3.0]
    ),
    wandb_dict=dict(
        project_name='ablation_study_graph'
    ),
    preprocess_dict=dict(
        thresh=1,
        translate=[1.5, 1.5, 1.5],
        p=0.75
    ),
    dcc_cutoff=1,
    # kfold_path="/cs/usr/punims/punims/MGClassifierV2/preprocessing/kfolds_chosen/training_list_seed_5069_cutoff_0.98_ban_0.56.txt",  # must be None if we want random.
    kfold_path="/cs/usr/punims/Desktop/punims-dinaLab/Databases/MG_Dummy_Graph_with_probes_DB/kfolds_dir/dummy_list.txt",  # must be None if we want random.
    radius=8,
    test=True,
    threshold=0.5,
    positive_label_threshold=3.0,
    ablation_index_to_zero=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    inference=False

)
