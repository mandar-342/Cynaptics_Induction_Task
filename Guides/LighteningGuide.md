***

# ⚡ Guide: Connecting Local VS Code to Lightning AI via SSH

## Step 1: Get Your Lightning AI Credentials

1. Open your browser, navigate to [Lightning AI](https://lightning.ai), and start your Studio.
2. On the right-hand toolbar, look for the **Terminal plugin** icon.
3. Click it and select **"Connect locally via SSH"**.
4. If this is your first time, click **"Setup new computer"**.
5. Lightning will display a command for your specific operating system (Mac/Linux or Windows). **Copy this command.** It will look something like this:
   `ssh <your-username>@ssh.lightning.ai`

## Step 2: Configure Your Local SSH Keys (One-Time Setup)

1. Open a terminal (or PowerShell on Windows) on your **local machine**.
2. **Paste and run the command** you copied from Lightning AI in Step 1. 
3. *Note:* If you don't already have an SSH key on your computer, you will need to generate one first. You can do this by running `ssh-keygen -t ed25519` and hitting Enter through the default prompts before running the Lightning command.
4. Running the Lightning command will automatically add your public key to the Studio's `authorized_keys`. 

## Step 3: Set Up VS Code

1. Open your local **VS Code**.
2. Go to the **Extensions** tab (`Ctrl+Shift+X` or `Cmd+Shift+X`).
3. Search for and install the **Remote - SSH** extension (published by Microsoft).

## Step 4: Make the Connection

1. In VS Code, open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`).
2. Type and select **`Remote-SSH: Connect to Host...`**
3. Select **`+ Add New SSH Host...`**
4. Paste your SSH command (`ssh <your-username>@ssh.lightning.ai`) and press Enter.
5. Select your default SSH configuration file (usually `~/.ssh/config` or `C:\Users\YourName\.ssh\config`).
6. A notification will pop up in the bottom right. Click **Connect**. (Alternatively, open the Command Palette again, select `Remote-SSH: Connect to Host...`, and click on `ssh.lightning.ai`).
7. If prompted about the fingerprint of the server, select **Continue**.

## Step 5: Open Your Project Directory
You are now connected to the remote machine, but you need to open your workspace.

1. In the remote VS Code window, go to the File Explorer and click **Open Folder**.
2. Type or select the path to your studio: `/teamspace/studios/this_studio`
3. Click **OK**.

---