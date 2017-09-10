import winsound
import tinytag
#winsound.Beep(frequency, duration)
winsound.Beep(250, 50000)
exit(0)
for i in range(200,220):
    print(i)
    winsound.Beep(i, 2000)
#from tinytag import TinyTag
#tag = TinyTag.get('rec1.mp3')
#print('This track is by %s.' % tag.artist)
#print('It is %f seconds long.' % tag.duration)`