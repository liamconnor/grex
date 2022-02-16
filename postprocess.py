import os 

import time
import numpy as np 

import coincidence

import slack

toslack = False

while True:
	outdir = coincidence.main(1)
	print(outdir)
	dsr = outdir.split('/')[-1]
	os.system('python /home/user/grex/make_cand_plots.py %s' % dsr)

	fnameout = '/home/user/grex/22011823/plots/59598.551784-station:3--dm17.7.png'
	if toslack:
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
        client.files_upload(channels='grex',file=fnameout,initial_comment=fnameout)


	print("Done coincidencing and plotting. Sleeping for a day now.")
	time.sleep(86400)