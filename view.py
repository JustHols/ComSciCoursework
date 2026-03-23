# view.py

# IMPORTS
import os
from threading import Thread
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import controller
import matplotlib.pyplot as plt

# SUBPROGRAMS
def help_popup() -> None:
    """
    Called when program starts and any time user presses
    'HELP' button to display help info on how to use app.
    """
    messagebox.showinfo("INSTRUCTIONS",
    "1) Press the 'File' button in the top menu to "
    "'Load' in a .wav file to edit and don't worry if you don't "
    "see anything happen, the file did load\n\n"
    "2) Add your desired effects using the "
    "'Effects' button in the top menu \n\n"
    "3) Modify each effect's settings using "
    "the sliders on each effect tile\n\n"
    "4) Use the other controls on each effect tile to delete "
    "the effect (the 'DEL' button), change the effect's "
    "position in the chain (the arrow buttons), and "
    "toggle the effect on and off (the checkbox)\n\n"
    "5) Play and stop the input and output audio samples "
    "using the 'PlayInput', 'PLayOutput', and 'STOP' buttons\n\n"
    "6) Use the 'LIVE' button to activate real time audio "
    "processing which can be stopped using the 'STOP' button\n\n"
    "7) When moving the sliders in live mode you can "
    "hear in real time how it changes the effect however in the "
    "normal mode you need to press 'STOP' and "
    "then 'PlayOutput' again to hear any changes\n\n"
    "8) Use the 'Presets' button to save and load "
    ".json files representing different effect combinations\n\n"
    "9) The 'QUIT' button can be used to exit the application\n\n"
    "10) You can access these instructions "
    "again by pressing the 'HELP' button\n\n"
    "11) Use the 'Plot' tile to view your selected "
    "sample's waveform, spectrum, and spectrogram using the "
    "corresponding buttons\n\n"
    "12) Use the 'Plot' tile's arrow buttons and then "
    "graphing buttons again to see how the graphs change after "
    "different effects are applied\n\n"
    "13) The graphs will be drawn on top of each other "
    "for easy comparison and cleared when you close the plot window")

def tile_warning_popup() -> None:
    """
    Called if user tries to add more Tile
    objects than the size of the grid to warn them.
    """
    messagebox.showwarning("WARNING",
    "The maximum number of effects has been reached. "
    "Please delete some effects in order to add more.")

def file_warning_popup() -> None:
    """
    Called if user tries to either play or plot a graph
    of a sample when no sample file has been loaded.
    """
    messagebox.showwarning("WARNING",
    "No audio file has been selected. "
    "Please load a .wav file and try again.")

# CLASSES
class View:
    """
    Class representing the view part of the
    model, view, controller structure. Provides
    a way for the user to interact with the application.

    Attributes:
        __controller (Controller): A controller object.
                                   Part of the MVC system.
        __root (tk.Tk): The root window of the application.

    Methods:
        run() -> None
    """
    def __init__(self) -> None:
        # ATTRIBUTES
        self.__controller = controller.Controller()
        self.__root = tk.Tk()

    # METHODS
    def run(self) -> None:
        """
        Called to start application.
        """
        # CONSTANTS
        # screen width and height
        WIDTH = 800
        HEIGHT = 400

        # MAIN
        # setting properties of root window
        self.__root.title("ECHO CHAMBER")
        self.__root.geometry(f"{WIDTH}x{HEIGHT}")
        self.__root.resizable(False, False)

        # instantiating MainFrame object and drawing it
        frame = MainFrame(self.__root,
                          WIDTH,
                          HEIGHT,
                          self.__controller)
        frame.pack()

        # running main loop waiting for GUI input
        self.__root.mainloop()

class MainFrame(tk.Frame):
    """
    Class representing the frame of the GUI in which
    all other frames sit inheriting from tk.Frame.

    Attributes:
        __root (tk.Tk): The root window.
        __controller (Controller): Controller in
                                   MVC architecture.
        __NUM_COLUMNS (int): Number of columns.
        __NUM_ROWS (int): Number of rows.
        __effects_info (dict[str: dict]): Effects information
                                          from model.
        __tiles (list[Tile]): List of all the
                              current Tile objects.
        __effect_tiles (list[EffectTile]): List of the current
                                           EffectTile objects.
        __filepath (str): The __filepath for the current
                          .wav file selected by the user.
        __graph_tile (GraphTile): GraphTile object.

    Methods:
        add_tile(name: str) -> None
        del_tile(tile: EffectTile) -> None
        shift_tile_left(tile: Tile) -> None
        shift_tile_right(tile: Tile) -> None
        play_input() -> None
        play_output() -> None
        stop_audio() -> None
        load_file() -> None
        save_file() -> None
        load_preset() -> None
        save_preset() -> None
        make_chain() -> dict[str, dict[str, float]]
        deconstruct_chain(chain: dict[str, dict[str, float]]) -> None
        process_chain() -> tuple[np.ndarray[np.float64], int]
        quit() -> None
        start_live_processing() -> None
        plot_waveform() -> None
        plot_spectrum() -> None
        plot_spectrogram() -> None
        process_graph_chain() -> tuple[np.ndarray[np.float64], int]
        make_graph_chain() -> dict[str, dict[str, float]]
    """
    def __init__(self,
                 root: tk.Tk,
                 width: int,
                 height: int,
                 controller: controller.Controller) -> None:
        super().__init__(root)

        # ATTRIBUTES
        self.__root = root
        self.__controller = controller
        self.__NUM_COLUMNS = 4
        self.__NUM_ROWS = 2
        self.__tiles = []
        self.__effect_tiles = []
        self.__graph_tile = None
        self.__filepath = None
        self.__effects_info = self.__controller.get_effects_info()

        # MAIN
        # setting row and column weights based on the desired number of tiles
        self.columnconfigure(list(range(self.__NUM_COLUMNS)),
                             weight=1,
                             minsize=width // self.__NUM_COLUMNS)
        self.rowconfigure(list(range(self.__NUM_ROWS)),
                          weight=1,
                          minsize=height // self.__NUM_ROWS)

        # creating a menu widget attached to the root window
        menubar = Menu(self.__root, self)
        self.__root.config(menu=menubar)

        # adding GraphTile
        self.__graph_tile = GraphTile(self, "#000000")
        self.__tiles.append(self.__graph_tile)
        self.update_tiles()

        # an instructions popup appears when the program is run
        help_popup()

    # METHODS
    def add_tile(self, name: str) -> None:
        """
        Adds new EffectTile object to self.__tiles and
        self.__effect_tiles. Updates all effect tiles.
        If max number of tiles reached displays warning.

        Args:
            name (str): EffectTile name.
        """
        if len(self.__tiles) >= self.__NUM_ROWS * self.__NUM_COLUMNS:
            tile_warning_popup()
        else:
            tile = EffectTile(self,
                              name,
                              self.__effects_info[name]["colour"],
                              self.__effects_info[name]["setup"])
            self.__tiles.append(tile)
            self.__effect_tiles.append(tile)
            self.update_tiles()

    def del_tile(self, tile: "EffectTile") -> None:
        """
        Removes an EffectTile object from self.__tiles and
        self.__effect_tiles before destroying widget.
        Updates all effect tiles.

        Args:
            tile (EffectTile): EffectTile object to be deleted.
        """
        self.__tiles.remove(tile)
        self.__effect_tiles.remove(tile)
        tile.destroy()
        self.update_tiles()

    def update_tiles(self) -> None:
        """
        For each Tile in self.__tiles redraws it
        to the grid based on index in self.__tiles.
        """
        for i, tile in enumerate(self.__tiles):
            tile.grid(row=i // self.__NUM_COLUMNS,
                      column=i % self.__NUM_COLUMNS,
                      sticky="news")

    def shift_tile_left(self, tile: "Tile") -> None:
        """
        Move Tile left by one in self.__tiles
        if possible then update all tiles.

        Args:
            tile (Tile): Tile object to be shifted.
        """
        index = self.__tiles.index(tile)

        # if tile first in list it can't shift left
        if index != 0:
            temp = self.__tiles[index]
            self.__tiles[index] = self.__tiles[index - 1]
            self.__tiles[index - 1] = temp

            self.update_tiles()

    def shift_tile_right(self, tile: "Tile") -> None:
        """
        Move Tile right by one in self.__tiles
        if possible then update all tiles.

        Args:
            tile (Tile): Tile object to be shifted.
        """
        index = self.__tiles.index(tile)

        # if tile last in list it can't shift right
        if index != len(self.__tiles) - 1:
            temp = self.__tiles[index]
            self.__tiles[index] = self.__tiles[index + 1]
            self.__tiles[index + 1] = temp

            self.update_tiles()

    def play_input(self) -> None:
        """
        Starts a thread to play the input audio
        file if the user has selected a file.
        If the user hasn't selected a file a
        warning is displayed.
        """
        if self.__filepath:
            Thread(target=lambda: self.__controller.play_file(self.__filepath)).start()
        else:
            file_warning_popup()

    def play_output(self) -> None:
        """
        If the user has selected a file it adds effects in
        the chain to sample and starts a thread to play data.
        If the user hasn't selected a file a
        warning is displayed.
        """
        if self.__filepath:
            data, samplerate = self.process_chain()
            Thread(target=lambda: self.__controller.play_data(data, samplerate)).start()
        else:
            file_warning_popup()

    def stop_audio(self) -> None:
        """
        Stops any audio playback including for live audio.
        """
        self.__controller.stop_audio()

    def load_file(self) -> None:
        """
        Allows user to select an input .wav file to load and stores its filepath.
        """
        self.__filepath = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")],
                                                     initialdir=os.path.join(os.getcwd(), "Samples"))

    def save_file(self) -> None:
        """
        Allows user to save input file with effects applied in
        desired location as a .wav file if there is an input
        file and location selected.
        """
        filepath = filedialog.asksaveasfilename(defaultextension=".wav",
                                                filetypes=[("WAV Files", "*.wav")],
                                                initialdir=os.path.join(os.getcwd(), "Samples"))
        if self.__filepath and filepath:
            data, samplerate = self.process_chain()
            self.__controller.save_file(filepath, data, samplerate)

    def load_preset(self) -> None:
        """
        Allows user to choose a preset .json file from file manager
        and then updates the effect tiles to reflect the
        chain described in the preset if they chose a preset.
        """
        filepath = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")],
                                              initialdir=os.path.join(os.getcwd(), "Presets"))
        if filepath:
            chain = self.__controller.load_preset(filepath)
            self.deconstruct_chain(chain)
            self.update_tiles()

    def save_preset(self) -> None:
        """
        User can save their current effects to a .json
        file specified in the file manager
        as a preset if they chose a file.
        """
        filepath = filedialog.asksaveasfilename(defaultextension=".json",
                                                filetypes=[("JSON Files", "*.json")],
                                                initialdir=os.path.join(os.getcwd(), "Presets"))
        chain = self.make_chain()
        if filepath and chain:
            self.__controller.save_preset(filepath, chain)

    def make_chain(self) -> dict[str, dict[str, float]]:
        """
        Creates an ordered dictionary called chain
        containing currently active effects and their settings.

        Returns:
            dict[str, dict[str, float]]: Chain of current effects.
        """
        chain = {}
        for tile in self.__effect_tiles:
            # checking if effect active
            if tile.get_state():
                chain[tile.get_name()] = tile.get_settings()
        return chain

    def deconstruct_chain(self, chain: dict[str, dict[str, float]]) -> None:
        """
        Called when loading a preset. Deletes all current tiles
        and then creates new tiles based on the effect chain
        described by preset.

        Args:
            chain (dict[str, dict[str, float]]):  Chain of effects
                                                  from preset.
        """
        # deleting all current effect tiles so
        # they can be replaced with preset effects
        for tile in self.__effect_tiles:
            self.del_tile(tile)

        # loop through each effect in chain and add
        # the new EffectTile to tiles list attributes
        for name, settings in chain.items():
            tile = EffectTile(self,
                              name,
                              self.__effects_info[name]["colour"],
                              self.__effects_info[name]["setup"])
            self.__tiles.append(tile)
            self.__effect_tiles.append(tile)

            # set value of scales in EffectTile frame based
            # off of the settings of effect in chain
            scales = tile.get_scales()
            labels = tile.get_labels()
            for i in range(len(scales)):
                scales[i].set(settings[labels[i]["text"]])

    def process_chain(self) -> tuple[np.ndarray[np.float64], int]:
        """
        Makes chain of effects currently active and applies them to
        the input file returning the processed data and samplerate.

        Returns:
            tuple[np.ndarray[np.float64], int]: Audio samplerate and data of
                                                input file with effects applied.
        """
        chain = self.make_chain()
        data, samplerate = self.__controller.process_chain(chain,
                                                           self.__filepath)
        return data, samplerate

    def quit(self) -> None:
        """
        Called to stop any currently playing audio
        and destroy root window of app when quitting.
        """
        self.stop_audio()
        self.__root.destroy()

    def start_live_processing(self) -> None:
        """
        When 'LIVE' button pressed called to start thread for live processing.
        """
        Thread(target=lambda: self.__controller.start_live_processing(self.make_chain)).start()

    def plot_waveform(self) -> None:
        """
        Plots the waveform of the audio sample at the
        point of the graph tile in the chain of effects if
        sample available to plot. If no sample selected a
        warning message will be displayed.
        """
        # if file has been selected get processed data and samplerate up to graph tile index
        if self.__filepath:
            data, samplerate = self.process_graph_chain()

            # process data for x and y axes
            xs = [i / samplerate for i in range(len(data))]

            # label plot
            plt.figure("WAVEFORM")
            plt.title("Waveform")
            plt.xlabel("time/s")
            plt.ylabel("amplitude")

            # display xs plotted against ys
            plt.plot(xs, data)
            plt.show()

        # waring message if no sample to graph
        else:
            file_warning_popup()

    def plot_spectrum(self) -> None:
        """
        Plots the spectrum of the audio sample at the
        point of the graph tile in the chain of effects if
        sample available to plot. If no sample selected a
        warning message will be displayed.
        """
        if self.__filepath:
            data, samplerate = self.process_graph_chain()

            # process data for x and y axes
            xs = np.fft.rfftfreq(len(data), 1 / samplerate)
            ys = np.abs(np.fft.rfft(data))

            # label plot
            plt.figure("SPECTRUM")
            plt.title("Spectrum")
            plt.xlabel("frequency/Hz")
            plt.ylabel("magnitude")

            # display xs plotted against ys
            plt.plot(xs, ys)
            plt.show()

        # waring message if no sample to graph
        else:
            file_warning_popup()

    def plot_spectrogram(self) -> None:
        """
        Plots the spectrogram of the audio sample at the
        point of the graph tile in the chain of effects if
        sample available to plot. If no sample selected a
        warning message will be displayed.
        """
        if self.__filepath:
            data, samplerate = self.process_graph_chain()

            # label plot
            plt.figure("SPECTROGRAM")
            plt.title("Spectrogram")
            plt.xlabel("time/s")
            plt.ylabel("frequency/Hz")

            # display xs plotted against ys
            plt.specgram(data, Fs=samplerate)
            plt.show()

        # waring message if no sample to graph
        else:
            file_warning_popup()

    def make_graph_chain(self) -> dict[str, dict[str, float]]:
        """
        Creates an ordered dictionary called chain containing
        currently active effects and their settings up to but
        not including location of graph tile in chain.

        Returns:
            dict[str, dict[str, float]]: Chain of effects
                                         up to GraphTile.
        """
        # getting index of GraphTile in __tiles
        index = self.__tiles.index(self.__graph_tile)

        chain = {}
        for tile in self.__effect_tiles[:index]:
            # checking if effect active
            if tile.get_state():
                chain[tile.get_name()] = tile.get_settings()
        return chain

    def process_graph_chain(self) -> tuple[np.ndarray[np.float64], int]:
        """
        Makes chain of effects currently active up to index of
        GraphTile and applies them to the input file returning
        the processed data and samplerate.

        Returns:
            tuple[np.ndarray[np.float64], int]: Audio samplerate and data of
                                                input file with effects applied.
        """
        chain = self.make_graph_chain()
        data, samplerate = self.__controller.process_chain(chain, self.__filepath)
        return data, samplerate

class Menu(tk.Menu):
    """
    Class representing menubar at top of the root window inheriting from tk.Menu.
    """
    def __init__(self, root: tk.Tk, master: MainFrame) -> None:
        super().__init__(root)

        # MAIN
        # adding submenu for file management
        file_menu = tk.Menu(self, tearoff=0)
        file_menu.add_command(label="Load",
                              command=master.load_file)
        file_menu.add_command(label="Save",
                              command=master.save_file)
        self.add_cascade(label="File", menu=file_menu)

        # adding submenu for managing presets
        presets_menu = tk.Menu(self, tearoff=0)
        presets_menu.add_command(label="Load",
                                 command=master.load_preset)
        presets_menu.add_command(label="Save",
                                 command=master.save_preset)
        self.add_cascade(label="Presets", menu=presets_menu)

        # creating submenu to add effects
        effects_menu = tk.Menu(self, tearoff=0)
        effects_menu.add_command(label="Lowpass",
                                 command=lambda: master.add_tile("LOWPASS"))
        effects_menu.add_command(label="Highpass",
                                 command=lambda: master.add_tile("HIGHPASS"))
        effects_menu.add_command(label="Bitcrusher",
                                 command=lambda: master.add_tile("BITCRUSHER"))
        effects_menu.add_command(label="Volume",
                                 command=lambda: master.add_tile("VOLUME"))
        effects_menu.add_command(label="PitchShifter",
                                 command=lambda: master.add_tile("PITCH SHIFTER"))
        effects_menu.add_command(label="Delay",
                                 command=lambda: master.add_tile("DELAY"))
        effects_menu.add_command(label="Flanger",
                                 command=lambda: master.add_tile("FLANGER"))
        self.add_cascade(label="Effects", menu=effects_menu)

        # adding buttons to top menubar which perform other functions
        self.add_command(label="PlayInput",
                         command=master.play_input)
        self.add_command(label="PlayOutput",
                         command=master.play_output)
        self.add_command(label="STOP",
                         command=master.stop_audio)
        self.add_command(label="LIVE",
                         command=master.start_live_processing)
        self.add_command(label="HELP",
                         command=help_popup)
        self.add_command(label="QUIT",
                         command=master.quit)

# abstract base class
class Tile(tk.Frame):
    """
    Superclass of GraphTile and EffectTile inheriting from tk.Frame
    to represent a tiles in the grid of tiles in MainFrame.

    Attributes:
        _left_button (tk.Button): Button to shift Tile left if possible.
        _right_button (tk.Button): Button to shift Tile right if possible.
    """
    def __init__(self, master: MainFrame, colour: str) -> None:
        super().__init__(master)
        # ATTRIBUTES
        # creating buttons for shifting tile left and right
        self._left_button = tk.Button(self,
                                      text="<-",
                                      command=lambda: master.shift_tile_left(self))
        self._right_button = tk.Button(self,
                                       text="->",
                                       command=lambda: master.shift_tile_right(self))

        # MAIN
        # adding a different coloured boarder around each Tile
        self.config(highlightbackground=colour,
                    highlightthickness=5)

class EffectTile(Tile):
    """
    Class inheriting from Tile to represent the effect
    tiles added to grid of tiles in MainFrame.

    Attributes:
        __name (str): The name of the effect tile.
        __state (tk.IntVar): The state of the tk.Checkbutton object.
        __labels (list[tk.Label]): A list of the tk.Label objects
                                   corresponding to the name of each effect setting.
        __scales (list[tk.Scale]): A list of the tk.Scale objects
                                   corresponding to each effect setting.

    Methods:
        get_settings() -> dict[str, float]
        get_state() -> bool
        get_name() -> str
        get_labels() -> list[tk.Label]
        get_scales() -> list[tk.Scale]
    """
    def __init__(self,
                 master: MainFrame,
                 name: str,
                 colour: str,
                 setup: dict[str, list[float]]) -> None:
        super().__init__(master, colour)

        # ATTRIBUTES
        self.__name = name
        self.__state = tk.IntVar(value=1)
        self.__labels = []
        self.__scales = []

        # MAIN
        # weighting rows and columns of the grid
        self.columnconfigure(list(range(3)), weight=1)
        self.rowconfigure(list(range(4)), weight=1)

        # drawing buttons to tile to delete and shift effect
        del_button = tk.Button(self,
                               text="DEL",
                               command=lambda: master.del_tile(self))
        self._left_button.grid(row=1, column=0, sticky="news")
        del_button.grid(row=1, column=1, sticky="news")
        self._right_button.grid(row=1, column=2, sticky="news")

        # adding checkbox to activate and deactivate the effect tile
        checkbutton = tk.Checkbutton(self,
                                     onvalue=1,
                                     offvalue=0,
                                     variable=self.__state)
        checkbutton.grid(row=0, column=2, sticky="news")

        # adding name of effect to tile
        name_label = tk.Label(self, text=self.__name)
        name_label.grid(row=0, column=0, columnspan=2, sticky="news")

        # creating scales and labels for each setting based
        # on effect_info about scale set up for each setting in effect
        for setting, setup in setup.items():
            self.__labels.append(tk.Label(self, text=setting))

            scale = tk.Scale(self,
                             from_=setup[1],
                             to=setup[0],
                             resolution=setup[2])
            scale.set(setup[3])

            self.__scales.append(scale)

        # drawing those scales and labels to tile frame grid
        for i in range(len(self.__scales)):
            self.__labels[i].grid(row=2, column=i, sticky="news")
            self.__scales[i].grid(row=3, column=i, sticky="news")

    # METHODS
    def get_settings(self) -> dict[str, float]:
        """
        Creates a dictionary for this tile effect containing
        the name of each setting and its current scale value.

        Returns:
            dict[str, float]: Effect settings for effect in effects info.
        """
        return {self.__labels[i]["text"]:
                    self.__scales[i].get() for i in range(len(self.__labels))}

    def get_state(self) -> bool:
        """
        Getter method for __state attribute.

        Returns:
            bool: Represents the state of the effect tile's checkbutton.
        """
        return bool(self.__state.get())

    def get_name(self) -> str:
        """
        Getter method for __name attribute.

        Returns:
            str: The name of the effect represented by EffectTile.
        """
        return self.__name

    def get_scales(self) -> list[tk.Scale]:
        """
        Getter method for __scales attribute.

        Returns:
             list[tk.Scale]: List of tk.Scale objects in EffectTile tk.Frame
        """
        return self.__scales

    def get_labels(self) -> list[tk.Label]:
        """
        Getter method for __labels attribute.

        Returns:
             list[tk.Label]: List of tk.Label objects in EffectTile tk.Frame
        """
        return self.__labels

class GraphTile(Tile):
    """
    Class inheriting from Tile providing user input
    to plot graph at point in chain shown by tile.
    """
    def __init__(self, master: MainFrame, colour: str) -> None:
        super().__init__(master, colour)

        # MAIN
        # weighting rows and columns of the grid
        self.columnconfigure(list(range(3)), weight=1)
        self.rowconfigure(list(range(4)), weight=1)

        # adding plot label
        plot_label = tk.Label(self, text="PLOT")
        plot_label.grid(sticky="news", row=0, column=1)

        # adding plotting buttons
        waveform_button = tk.Button(self,
                                    text="waveform",
                                    command=master.plot_waveform)
        spectrum_button = tk.Button(self,
                                    text="spectrum",
                                    command=master.plot_spectrum)
        spectrogram_button = tk.Button(self,
                                       text="spectrogram",
                                       command=master.plot_spectrogram)
        waveform_button.grid(sticky="news", columnspan=3, row=1, column=0)
        spectrum_button.grid(sticky="news", columnspan=3, row=2, column=0)
        spectrogram_button.grid(sticky="news", columnspan=3, row=3, column=0)

        # drawing shifting buttons
        self._left_button.grid(row=0, column=0, sticky="news")
        self._right_button.grid(row=0, column=2, sticky="news")