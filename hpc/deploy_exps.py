import subprocess

cfgs = [
    ('s2s-exp-oaout.yaml', 's2s-def.yaml'),
    ('s2s-exp-oaobs.yaml', 's2s-def.yaml'),
    ('s2s-exp-bgsint.yaml', 's2s-def.yaml'),
    ('s2s-exp-bgsobs.yaml', 's2s-def.yaml'),
    ('s2s-exp-oaout.yaml', 's2s-def-async-up.yaml'),
    ('s2s-exp-oaobs.yaml', 's2s-def-async-up.yaml'),
    ('s2s-exp-bgsint.yaml', 's2s-def-async-up.yaml'),
    ('s2s-exp-bgsobs.yaml', 's2s-def-async-up.yaml'),
    ('s2s-exp-oaout.yaml', 's2s-def-async-weight.yaml'),
    ('s2s-exp-oaobs.yaml', 's2s-def-async-weight.yaml'),
    ('s2s-exp-bgsint.yaml', 's2s-def-async-weight.yaml'),
    ('s2s-exp-bgsobs.yaml', 's2s-def-async-weight.yaml'),
    ('t2t-exp-s1.yaml', 't2t-def.yaml'),
    ('t2t-exp-s2.yaml', 't2t-def.yaml'),
    ('t2t-exp-s3.yaml', 't2t-def.yaml'),
    ('t2t-exp-s1.yaml', 't2t-def-async-up.yaml'),
    ('t2t-exp-s2.yaml', 't2t-def-async-up.yaml'),
    ('t2t-exp-s3.yaml', 't2t-def-async-up.yaml'),
    ('t2t-exp-s1.yaml', 't2t-def-async-weight.yaml'),
    ('t2t-exp-s2.yaml', 't2t-def-async-weight.yaml'),
    ('t2t-exp-s3.yaml', 't2t-def-async-weight.yaml'),
]
base_cmds = ['qsub', 'start_worker_for_exps.sh']

for c in cfgs:
    cfg_cmd = ['-F configs/exps/{} configs/exps/{}'.format(*c)]
    cmd = base_cmds + cfg_cmd
    print(cmd)
    subprocess.run(cmd)