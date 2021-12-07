#!/usr/bin/env bash

rsync -avz --exclude data --exclude work_dirs --exclude tmp --exclude mmsegmentation.egg-info ~/code/spacenet6 sftp_whuhpc:/home/chenshuailin/code/