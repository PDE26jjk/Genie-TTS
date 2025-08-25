import argparse
import shlex
from rich.table import Table
from typing import Optional, Callable
from .Audio.ReferenceAudio import ReferenceAudio
from .Utils.Shared import console, context
from .Utils.UserData import userdata_manager
from .ModelManager import model_manager
from .Core.TTSPlayer import tts_player


class Client:
    def __init__(self):
        self.commands: dict[str, Callable] = {
            'load': self._handle_load,
            'unload': self._handle_unload,
            'speaker': self._handle_speaker,
            'prompt': self._handle_prompt,
            'say': self._handle_say,
            'stop': self._handle_stop,
            'help': self._handle_help,
        }

    def _handle_load(self, args_list: list):
        """
        加载角色模型（如果省略 model_path，将使用上次的路径）。
        用法: /load <character_name> <model_path>
        """
        parser = argparse.ArgumentParser(prog="/load", description=self._handle_load.__doc__.strip())
        parser.add_argument('character', help='要加载的角色的名字。')
        parser.add_argument('path', nargs='?', default=None,
                            help='角色模型所在的目录路径，若有反斜杠请用双引号括起来。(可选)')

        try:
            args = parser.parse_args(args_list)
            model_path: Optional[str] = args.path
            all_cached_paths: dict[str, str] = userdata_manager.get('last_model_paths', {})

            # 如果用户未直接提供路径，则尝试从缓存加载
            if model_path is None:
                if not all_cached_paths or args.character not in all_cached_paths:
                    console.print("[bold red]错误：[/bold red]您未输入模型文件夹路径。")
                    return
                model_path = all_cached_paths[args.character]
                console.print(f"未提供路径，使用缓存路径: [green]{model_path}[/green]")

            # 验证成功，执行加载并更新缓存
            model_manager.load_character(character_name=args.character, model_dir=model_path)
            all_cached_paths[args.character] = model_path
            userdata_manager.set('last_model_paths', all_cached_paths)
            console.print(f"成功加载角色 '{args.character}'！")

        except SystemExit:
            pass  # 捕获 argparse 的 -h 或错误，防止程序退出
        except Exception as e:
            console.print(f"[bold red]加载时发生未知错误：[/bold red] {e}")

    def _handle_unload(self, args_list: list):
        """
        卸载角色模型，释放资源。
        用法: /unload <character_name>
        """
        parser = argparse.ArgumentParser(prog="/unload", description=self._handle_unload.__doc__.strip())
        parser.add_argument('character', help='要卸载的角色的名字。')
        try:
            args = parser.parse_args(args_list)
            model_manager.remove_character(character_name=args.character)
            console.print(f"角色 '{args.character}' 已卸载。")
        except SystemExit:
            pass

    def _handle_speaker(self, args_list: list):
        """
        切换当前说话人。
        用法: /speaker <character_name>
        """
        parser = argparse.ArgumentParser(prog="/speaker", description=self._handle_speaker.__doc__.strip())
        parser.add_argument('character', help='要切换到的角色的名字。')
        try:
            args = parser.parse_args(args_list)
            if not model_manager.has_character(args.character):
                console.print("[bold red]错误：[/bold red]该角色不存在，请先导入角色。")
                return
            context.current_speaker = args.character
            console.print(f"当前说话人已设置为 '{args.character}'。")
        except SystemExit:
            pass

    def _handle_prompt(self, args_list: list):
        """
        设置参考音频和文本。
        用法: /prompt <audio_path> <text>
        """
        parser = argparse.ArgumentParser(prog="/prompt", description=self._handle_prompt.__doc__.strip())
        parser.add_argument('audio_path', help='参考音频的路径。')
        parser.add_argument('text', help='参考音频对应的文本。')
        try:
            args = parser.parse_args(args_list)
            context.current_prompt_audio = ReferenceAudio(prompt_wav=args.audio_path, prompt_text=args.text)
            console.print("参考音频设置成功。")
        except SystemExit:
            pass

    def _handle_say(self, args_list: list):
        """
        文本到语音合成。
        用法: /say <text_to_say> [-o/--output path] [--play]
        """
        parser = argparse.ArgumentParser(prog="/say", description=self._handle_say.__doc__.strip())
        parser.add_argument('text', help='要转换为语音的文本。')
        parser.add_argument('-o', '--output', help='将音频保存到的文件路径。(可选)')
        parser.add_argument('--play', action='store_true', help='播放生成的音频。(可选)')
        try:
            args = parser.parse_args(args_list)
            tts_player.start_session(
                play=args.play,
                save_path=args.output
            )
            tts_player.feed(args.text)
            tts_player.end_session()
            tts_player.wait_for_tts_completion()
        except SystemExit:
            pass

    @staticmethod
    def _handle_stop(args_list: list):
        """
        停止所有当前和待处理的任务。
        """
        try:
            tts_player.stop()
            console.print("所有任务已停止。")
        except SystemExit:
            pass

    def _handle_help(self, args_list: list):
        """
        获取所有指令的帮助信息。
        """
        console.print("\n可用指令:", justify="left")

        table = Table(box=None, show_header=False, pad_edge=False)
        table.add_column("Command", style="bold cyan", width=15)
        table.add_column("Description")

        for cmd, handler in self.commands.items():
            doc = handler.__doc__
            if not doc:
                description = "[italic]无描述[/italic]"
            else:
                # 清理并分割文档字符串
                doc_lines = [line.strip() for line in doc.strip().split('\n')]
                description = "\n".join(doc_lines) + "\n"

            table.add_row(f"/{cmd}", description)

        console.print(table)

    def run(self):
        """
        启动客户端的交互式主循环。
        """
        console.print(
            "欢迎使用 [bold cyan]GENIE[/bold cyan] 命令行。输入 [bold blue]/help[/bold blue] 获取帮助，按下 Ctrl+C 退出。")

        while True:
            try:
                raw_input = console.input("[bold]>> [/bold]")

                if not raw_input:
                    continue
                if not raw_input.startswith('/'):
                    console.print("[bold red]错误：[/bold red]指令必须以 '/' 开头。输入 /help 查看帮助。")
                    continue

                parts = shlex.split(raw_input[1:])
                if not parts:
                    continue

                command_name = parts[0].lower()
                command_args = parts[1:]

                handler = self.commands.get(command_name)
                if handler:
                    handler(command_args)
                else:
                    console.print(f"[bold red]错误：[/bold red]未知指令 '[yellow]/{command_name}[/yellow]'。")

            except (KeyboardInterrupt, EOFError):
                break
