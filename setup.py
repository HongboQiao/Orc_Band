import sys
sys.path.insert(0, "Organic-Semi/Documents")
import SMS_BandGap

sms = input("Please Input a SMILES String: ")

BG = SMS_BandGap.sms_bandgap(sms)

print("The Predictied Band Gap is: ", BG)
EXIT01 = input('Please Press Enter to Exit')
