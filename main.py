from controller import Controller
from audioplayer import AudioPlayer

if __name__ == '__main__':
    player = AudioPlayer('ateamtheme.mp3')
    player.play()
    Controller().run()
