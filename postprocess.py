import os 

import time
import numpy as np 
import glob

import coincidence

#import slack

to_slack = False
to_email = True 
mailer_list = ['liam.dean.connor@gmail.com', 'kshila@caltech.edu']

if to_email:
    import mailer 

while True:
    outdir = coincidence.main(4)
    print(outdir)
    dsr = outdir.split('/')[-1]
    os.system('python make_cand_plots.py %s' % dsr)

#    fn_cand_plots = filter(os.path.isdir, glob.glob('/home/user/grex/2*'))
#    fn_cand_plots = sorted(fn_cand_plots, key=os.path.getmtime)  
    fn_cand_plots = glob.glob(outdir+'/plots/*png')
    
    if to_slack:
        print("Sending to slack")
        slack_file = '{0}/.config/slack_api'.format(
            os.path.expanduser("~")
        )
        if not os.path.exists(slack_file):
            raise RuntimeError(
                "Could not find file with slack api token at {0}".format(
                    slack_file
                    )
            )
        with open(slack_file) as sf_handler:
            slack_token = sf_handler.read()
        client = slack.WebClient(token=slack_token)
        client.files_upload(channels='grex',file=fn_cand_plots,initial_comment=fnameout)

    if to_email:
        mailer.send_email(send_from='grexalerts@gmail.com', 
                          send_to=mailer_list, 
                subject="GReX Candidates", text="%d candidates"%len(fn_cand_plots), files=fn_cand_plots, 
               server = "smtp.gmail.com", 
               port = 587, passwd='rqqddxnuiuozjnah')


    print("Done coincidencing and plotting. Sleeping for a day now.")
    time.sleep(86400)
