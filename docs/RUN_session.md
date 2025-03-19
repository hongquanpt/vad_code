# 1. tmux
## To install tmux on Debian/Ubuntu, enter the following:
```bash
sudo apt install tmux
```

## start new session
```bash
tmux new -s myname
tmux new -s pbmae
```

## Detaching from Tmux Session
```bash
Ctrl + B D
```

## attach to name
```bash
tmux a -t myname
tmux a -t pbmae
```

## list session
```bash
tmux ls
```

## kill session
```bash
tmux kill-session -t myname
or in a specified tmux session, typing command "exit"
```

## To close all tmux sessions I use:
```bash
tmux kill-session
``` 

## Scroll in tmux
```bash
- Ctrl-b then [ then you can use your normal navigation keys to scroll around (eg. Up Arrow or PgDn). 
- Press q to quit scroll mode.
```

# 2. screen
## To install Screen on Debian/Ubuntu, enter the following:
```bash
sudo apt install screen
```

## To launch Linux Screen and start a screen session, run the command:
```bash
screen
Press Space to continue to the next page.
Press Space again to open a new screen session.
To see a list of available commands press the keys Ctrl + a, followed by ?.
```

## To launch and name a new session, use the command:
```bash
screen -S session_name
```

## The most commonly used keystrokes include:
```bash
Ctrl + a and c – Open a new screen window.
Ctrl + a and " – List all open windows.
Ctrl + a and 0 – Switch to window 0 (or any other numbered window).
Ctrl + a and A – Rename the current window.
Ctrl + a and S - Split the screen horizontally, with the current window on top.
Ctrl + a and | - Split the screen vertically, with the current window on the left.
Ctrl + a and tab – Switch focus between areas of the split screen.
Ctrl + a and Ctrl + a – Switch between current and previous windows.
Ctrl + a and n – Switch to the next window.
Ctrl + a and p – Switch to the previous window.
Ctrl + a and Q – Quit all other windows except the current one..
Ctrl + a and X – Lock the current window.
Ctrl + a and H – Create a running log of the session.
Ctrl + a and M – Monitor a window for output (a notification pops up when that window has activity).
Ctrl + a and _ - Watch a window for absence of output (such as when a file finishes downloading or a compiler finishes).
```

## Detaching and Reattaching Screen
- To detach from screen and leave the window running in the background, use the keystroke:
```bash
Ctrl + a and d
```

- To reattach to a running screen session, use:
```bash
screen -r
```

- Each screen session has a different ID and you can see the session ID list with the command:
```bash
screen -ls
```

- Once you have the ID, add it to the screen -r command:
```bash
screen -r sessionID
```