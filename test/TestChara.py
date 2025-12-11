import genie_tts as genie


def test_feibi():
    genie.load_predefined_character('feibi')
    genie.tts(
        character_name='feibi',
        text='棉花大哥哥和鬼蜀黍真是一对苦命鸳鸯啊。',
        play=True,
    )
    genie.wait_for_playback_done()


def test_mika():
    genie.load_predefined_character('mika')
    genie.tts(
        character_name='mika',
        text='今日も素敵な一日をお過ごしください。',
        play=True,
    )
    genie.wait_for_playback_done()


def test_37():
    genie.load_predefined_character('thirtyseven')
    genie.tts(
        character_name='thirtyseven',
        text='Hello! Welcome to our language learning platform.',
        play=True,
    )
    genie.wait_for_playback_done()


if __name__ == '__main__':
    test_37()
