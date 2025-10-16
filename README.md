Telemuze

What it is
- Telegram bot that transcribes audio/video to plain text using WhisperX.
- Architecture: one long-lived listener VM (bot) provisions short-lived composer VMs to do CPU-only transcription over SSH.

## Docker images

There are two Dockerfiles, one for each role. These can be used for local testing or for deployment to the ThreeFold Grid as flists.

Here's an example of building the containers for local use. To push them remotely, you'll need to adjust the tags according to your username on Docker Hub, for example.

- Listener
```
docker buildx build -t telemuze-listener -f listener/Dockerfile .
```
- Composer:
```
docker buildx build -t telemuze-listener -f composer/Dockerfile .
```

## Deploy the listener VM using tfcmd (CLI)

Prereqs
- You have `tfcmd` configured with your `TF_MNEMONIC` and network access.
- Your base image starts zinit and sshd automatically.

Minimal zinit service on the VM
- File: `/etc/zinit/telemuze-listener.yaml`
- Content:
  - name: telemuze-listener
    exec: ["python3", "-m", "listener.main"]
    env: /etc/zinit/env/telemuze-listener.env

Minimal env file on the VM
- File: `/etc/zinit/env/telemuze-listener.env`
- Required:
  - `TELEGRAM_BOT_TOKEN=123:ABC`
  - `TF_MNEMONIC="your mnemonic words ..."` (used by the listener to provision composers)
- Optional (recommended defaults shown):
  - `TF_NETWORK=main`
  - `DEFAULT_MODEL=large-v3`
  - `DEFAULT_LANGUAGE=auto`
  - `ALLOWED_USERNAMES=alice,bob` (empty = allow all)
  - `ALLOWED_USER_IDS=`
  - `MAX_COMPOSERS=3`
  - `PER_USER_CONCURRENCY=1`
  - `JOB_TIMEOUT_SEC=10800`
  - `CACHE_WARM_INTERVAL_MIN=360`
  - `COMPOSER_USERNAME=transcriber`
  - `TF_NODE_ID=` and/or `TF_FARM_ID=`

- Deploy with tfcmd (fill in flags to match your tfcmd version):
  - tfcmd vm deploy \
      --name telemuze-listener \
      --image registry.example.com/telemuze-listener:latest \
      --cpu 2 \
      --mem 2048 \
      --disk 10 \
      --farm-id <FARM_ID> \    # optional
      --node-id <NODE_ID> \    # optional
      --user-data @cloud-init-listener.yaml

Option B: deploy then copy files
- Deploy the VM image:
  - tfcmd vm deploy --name telemuze-listener --image registry.example.com/telemuze-listener:latest --cpu 2 --mem 2048 --disk 10 [--farm-id ...] [--node-id ...]
- Get the VM IP:
  - tfcmd vm ip --name telemuze-listener
- SSH in and create:
  - /etc/zinit/env/telemuze-listener.env (as above)
  - /etc/zinit/telemuze-listener.yaml (as above)
- Start:
  - zinit init && zinit update && zinit start telemuze-listener

3) Local testing with Docker (no Grid)

Goal
- Run a composer container with sshd.
- Run the listener locally and point it at the composer via `COMPOSER_IP_OVERRIDE`.

Steps
- Build:
  - docker build -f Dockerfile.listener -t telemuze-listener:dev .
  - docker build -f Dockerfile.composer -t telemuze-composer:dev .
- Start composer:
  - docker network create telemuze-net || true
  - docker run -d --name telemuze-composer --network telemuze-net telemuze-composer:dev
- Authorize the listener’s SSH key on the composer:
  - mkdir -p ~/.telemuze
  - [ -f ~/.telemuze/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.telemuze/id_ed25519 -C telemuze
  - docker exec -u root telemuze-composer bash -lc 'useradd -ms /bin/bash transcriber || true; install -d -m 700 ~transcriber/.ssh'
  - docker cp ~/.telemuze/id_ed25519.pub telemuze-composer:/tmp/id.pub
  - docker exec -u root telemuze-composer bash -lc 'cat /tmp/id.pub >> ~transcriber/.ssh/authorized_keys; chown -R transcriber:transcriber ~transcriber/.ssh; chmod 600 ~transcriber/.ssh/authorized_keys'
- Get composer IP:
  - COMPOSER_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' telemuze-composer)
- Run listener:

```bash
docker run --rm -it --name telemuze-listener \
  -e TELEGRAM_BOT_TOKEN="123:ABC" \
  -e TF_MNEMONIC="local-testing" \
  -e TF_NETWORK="main" \
  -e DEFAULT_MODEL="large-v3" \
  -e DEFAULT_LANGUAGE="auto" \
  -e ALLOWED_USERNAMES="your_tg_username" \
  telemuze-listener
  ```

Send a voice/audio/video to your bot on Telegram; you should receive a transcript reply.

Notes
- Replace image names, resources, and tfcmd flags to match your environment.
- The listener will still try to provision composers via tfcmd; for local-only testing, set `COMPOSER_IP_OVERRIDE` and, if needed, stub the provisioning step in `listener/main.py` to skip tfcmd when the override is present.
