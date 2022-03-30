import os
import subprocess
import numpy as np
import datetime
import time
import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
import smtplib, ssl
#from astropy.constants import c
#import astropy.constants as const
#from astropy.time import Time
#from astropy.coordinates import SkyCoord,EarthLocation,AltAz,Angle,ICRS,ITRS,get_icrs_coordinates
#from astropy.wcs import WCS
#import astropy.units as u

def send_email(send_from='grexalerts@gmail.com', send_to=['liam.dean.connor@gmail.com'], 
			   subject="GReX Candidates", text="", files=[], 
			   server = "smtp.gmail.com", 
			   port = 587, passwd='rqqddxnuiuozjnah'):
	assert isinstance(send_to, list)
	msg = MIMEMultipart()
	msg['From'] = send_from
	msg['To'] = COMMASPACE.join(send_to)
	msg['Date'] = formatdate(localtime=True)
	msg['Subject'] = subject
	msg.attach(MIMEText(text))
	for f in files or []:
		with open(f,"rb") as fil:
			part = MIMEApplication(
				fil.read(),
				Name=basename(f)
			)
		part['Content-Disposition'] = 'attachment; filename="%s"' %basename(f)
		msg.attach(part)
	smtp = smtplib.SMTP(server,port)
	smtp.set_debuglevel(1)
	smtp.ehlo()
	smtp.starttls()
	smtp.ehlo()
	smtp.login(send_from,passwd)
	smtp.sendmail(send_from,send_to,msg.as_string())
	smtp.close()