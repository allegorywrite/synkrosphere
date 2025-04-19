"""
映像レンダリングモジュール
VAEの出力とGLSLシェーダーを組み合わせて映像を生成する
"""

import numpy as np
import moderngl
import pygame
from pygame.locals import *
import logging

logger = logging.getLogger(__name__)

class Renderer:
    """
    映像レンダリングクラス
    VAEの出力画像とGLSLシェーダーを組み合わせて映像を生成する
    """
    
    def __init__(self, width=1920, height=1080, fullscreen=False):
        """
        Args:
            width: 出力映像の幅
            height: 出力映像の高さ
            fullscreen: フルスクリーンモードで起動するかどうか
        """
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        
        pygame.init()
        flags = DOUBLEBUF | OPENGL
        if fullscreen:
            flags |= FULLSCREEN
        
        self.screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption("Synkrosphere")
        
        self.ctx = moderngl.create_context()
        
        self.programs = {}
        self._init_shaders()
        
        self.textures = {}
        self._init_textures()
        
        self.quad = self._create_quad()
        
        logger.info(f"Renderer initialized: {width}x{height}, fullscreen={fullscreen}")
    
    def _init_shaders(self):
        """シェーダープログラムの初期化"""
        vertex_shader = """
            in vec2 in_position;
            in vec2 in_texcoord;
            out vec2 uv;
            void main() {
                gl_Position = vec4(in_position, 0.0, 1.0);
                uv = in_texcoord;
            }
        """
        
        fragment_shader = """
            uniform sampler2D texture0;
            uniform float time;
            in vec2 uv;
            out vec4 fragColor;
            void main() {
                fragColor = texture(texture0, uv);
            }
        """
        
        self.programs['basic'] = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
    
    def _init_textures(self):
        """テクスチャの初期化"""
        self.textures['vae_output'] = self.ctx.texture(
            (self.width, self.height), 3, dtype='f4'
        )
        self.textures['vae_output'].filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        self.textures['feedback'] = self.ctx.texture(
            (self.width, self.height), 3, dtype='f4'
        )
        self.textures['feedback'].filter = (moderngl.LINEAR, moderngl.LINEAR)
    
    def _create_quad(self):
        """レンダリング用の四角形を作成"""
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0,
        ], dtype='f4')
        
        indices = np.array([
            0, 1, 2,
            0, 2, 3,
        ], dtype='i4')
        
        vbo = self.ctx.buffer(vertices)
        ibo = self.ctx.buffer(indices)
        
        return self.ctx.vertex_array(
            self.programs['basic'],
            [
                (vbo, '2f 2f', 'in_position', 'in_texcoord'),
            ],
            ibo
        )
    
    def update_vae_texture(self, image_data):
        """VAE出力テクスチャを更新"""
        self.textures['vae_output'].write(image_data)
    
    def render(self, shader_name='basic', uniforms=None):
        """
        シーンをレンダリング
        
        Args:
            shader_name: 使用するシェーダー名
            uniforms: シェーダーに渡す追加のuniform変数
        """
        if shader_name not in self.programs:
            logger.warning(f"Shader '{shader_name}' not found, using 'basic'")
            shader_name = 'basic'
        
        program = self.programs[shader_name]
        
        if uniforms:
            for name, value in uniforms.items():
                if name in program:
                    program[name] = value
        
        program['time'] = pygame.time.get_ticks() / 1000.0
        
        self.textures['vae_output'].use(0)
        program['texture0'] = 0
        
        self.quad.render(moderngl.TRIANGLES)
        
        pygame.display.flip()
    
    def handle_events(self):
        """イベント処理"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
        return True
    
    def cleanup(self):
        """リソースの解放"""
        for texture in self.textures.values():
            texture.release()
        
        for program in self.programs.values():
            program.release()
        
        self.quad.release()
        pygame.quit()
        
        logger.info("Renderer resources cleaned up")
