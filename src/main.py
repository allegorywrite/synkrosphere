"""
Synkrosphere - AIによるリアルタイムVJ/DJシステム
メインエントリーポイント
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description='Synkrosphere - AIによるリアルタイムVJ/DJシステム')
    parser.add_argument('--config', type=str, default='config/default.yaml', 
                        help='設定ファイルのパス')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='inference',
                        help='実行モード: 学習またはリアルタイム推論')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='使用デバイス (cuda/cpu)')
    return parser.parse_args()

def main():
    """メイン実行関数"""
    args = parse_args()
    logger.info(f"Synkrosphere を起動中... モード: {args.mode}")
    
    
    if args.mode == 'train':
        logger.info("学習モードで起動します")
    else:
        logger.info("推論モードで起動します")
    
    logger.info("Synkrosphere を終了します")

if __name__ == "__main__":
    main()
