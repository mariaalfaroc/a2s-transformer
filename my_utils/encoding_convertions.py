import re
from typing import List, Union

import numpy as np

VOICE_CHANGE_TOKEN = "<COC>"
STEP_CHANGE_TOKEN = "<COR>"


class krnParser:
    """Main Kern parser operations class."""

    def __init__(self, use_voice_change_token: bool = True) -> None:
        self.reserved_words = ["clef", "k[", "*M"]
        self.reserved_dot = "."
        self.reserved_dot_EncodedCharacter = "DOT"
        self.clef_change_other_voices = "*"
        self.comment_symbols = ["*", "!"]
        self.voice_change = VOICE_CHANGE_TOKEN  # change-of-column (coc) token
        self.step_change = STEP_CHANGE_TOKEN  # change-of-row (cor) token
        self.use_voice_change_token = use_voice_change_token

    # ---------------------------------------------------------------------------- AUXILIARY FUNCTIONS

    def _readSrcFile(self, text: str) -> np.ndarray:
        """Adequate a Kern file content to the correct format for further processes."""
        in_src = text.splitlines()

        # Locating line with the headers
        it_headers = 0
        while "**kern" not in in_src[it_headers]:
            it_headers += 1
        pass
        columns_to_process = np.where(np.array(in_src[it_headers].split("\t")) == "**kern")[0]

        # Locating lines with comments (to be removed)
        in_src_nocomments = []
        for line in in_src:
            if not line.strip().startswith("!"):
                in_src_nocomments.append(line.split("\t"))
        pass

        # Extract voices and removing lines with comments
        out_src = np.array(in_src_nocomments)[:, columns_to_process]

        return out_src

    def _postprocessKernSequence(self, in_score: np.ndarray) -> np.ndarray:
        """Exchanging '*' for the actual symbol."""

        # Retrieving positions with '*'
        positions = sorted(list(set(np.where(in_score == "*")[1])))

        # For each position,
        # we retrieve the last explicit clef symbol and include it in the stream
        for single_position in positions:
            for it_voice in range(in_score.shape[0]):
                if in_score[it_voice, single_position] == "*":
                    new_element = in_score[
                        it_voice,
                        max(np.where(np.char.startswith(in_score[it_voice], "*clef"))[0]),
                    ]
                    in_score[it_voice, single_position] = new_element
                pass
            pass
        pass

        return in_score

    def cleanKernFile(self, text: str) -> np.ndarray:
        """Convert complete kern sequence to CLEAN kern format."""
        in_file = self._readSrcFile(text=text)

        # Processing individual voices
        out_score = []
        for it_voice in range(in_file.shape[1]):
            in_voice = in_file[:, it_voice].tolist()
            out_voice = [self.cleanKernToken(u) for u in in_voice if self.cleanKernToken(u) is not None]

            out_score.append(out_voice)
        pass
        out_score = np.array(out_score)

        # Postprocess obtained score
        out_score = self._postprocessKernSequence(out_score)

        return out_score

    def cleanKernToken(self, in_token: str) -> Union[str, None]:
        """Convert a kern token to its CLEAN equivalent."""
        out_token = None  # Default

        if any([u in in_token for u in self.reserved_words]):  # Relevant reserved tokens
            out_token = in_token

        elif in_token == self.reserved_dot:  # Case when using "." for sync. voices
            out_token = self.reserved_dot_EncodedCharacter

        elif in_token.strip() == self.clef_change_other_voices:  # Clef change in other voices
            out_token = in_token

        elif any([in_token.startswith(u) for u in self.comment_symbols]):  # Comments
            out_token = None

        elif in_token.startswith("s"):  # Slurs
            out_token = "s"

        elif "=" in in_token:  # Bar lines
            out_token = "="

        elif not "q" in in_token:
            if "rr" in in_token:  # Multirest
                out_token = re.findall(r"rr[0-9]+", in_token)[0]
            elif "r" in in_token:  # Rest
                out_token = in_token.split("r")[0] + "r"
            else:  # Music note
                out_token = re.findall(r"\[*\d+[.]*[a-gA-G]+[n#-]*\]*", in_token)[0]

        return out_token

    # ---------------------------------------------------------------------------- CONVERT CALL

    def convert(self, text: str) -> List[str]:
        out = self.cleanKernFile(text).T

        out_line = []
        for t in out:
            for v in t:
                out_line.append(v)
                if self.use_voice_change_token:
                    out_line.append(self.voice_change)
            if self.use_voice_change_token:
                del out_line[-1]
            out_line.append(self.step_change)
        del out_line[-1]

        return out_line
