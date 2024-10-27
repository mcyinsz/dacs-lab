import os
from typing import Callable

from .stdcell_library import StdcellLibrary

def collect_filtered_files(root: str, filter_func: Callable) -> list:
    """
        Collect files from root directory using filter_func and output_func.
    """
    files = os.listdir(root)
    files = list(filter(filter_func, files))
    files = [os.path.join(root, f) for f in files]
    return files

class Sky130Library(StdcellLibrary):
    """
        Sky130 open-sourced PDK
    """

    def __init__(
        self, 
        pdk_dir: str, 
        syn_tool: str = 'genus',
        pnr_tool: str = 'innovus',
    ) -> None:
        super().__init__(pdk_dir, syn_tool, pnr_tool)
        assert syn_tool == 'genus', "We cannot support Yosys for now"
        assert pnr_tool == 'innovus', "We cannot support OpenROAD for now"

    @property
    def name(self) -> str:
        return "Sky130"

    @property
    def lib_files(self) -> list:
        lib_files = [os.path.join(self.pdk_dir, 'lib', 'sky130_fd_sc_hd__tt_025C_1v80.lib')]
        return lib_files

    @property
    def lef_files(self) -> list:
        lef_files = [
            # techlef
            os.path.join(self.pdk_dir, 'lef', 'sky130_fd_sc_hd.tlef'),
            # stdcell lef
            os.path.join(self.pdk_dir, 'lef', 'sky130_fd_sc_hd_merged.lef'),
        ]
        # stdcell lef
        return lef_files
    
    @property
    def dont_use_cells(self) -> list:
        return [
            "sky130_fd_sc_hd__probec_p_8",
            "sky130_fd_sc_hd__lpflow_bleeder_1",
            "sky130_fd_sc_hd__lpflow_clkbufkapwr_1",
            "sky130_fd_sc_hd__lpflow_clkbufkapwr_16",
            "sky130_fd_sc_hd__lpflow_clkbufkapwr_2",
            "sky130_fd_sc_hd__lpflow_clkbufkapwr_4",
            "sky130_fd_sc_hd__lpflow_clkbufkapwr_8",
            "sky130_fd_sc_hd__lpflow_clkinvkapwr_1",
            "sky130_fd_sc_hd__lpflow_clkinvkapwr_16",
            "sky130_fd_sc_hd__lpflow_clkinvkapwr_2",
            "sky130_fd_sc_hd__lpflow_clkinvkapwr_4",
            "sky130_fd_sc_hd__lpflow_clkinvkapwr_8",
            "sky130_fd_sc_hd__lpflow_decapkapwr_12",
            "sky130_fd_sc_hd__lpflow_decapkapwr_3",
            "sky130_fd_sc_hd__lpflow_decapkapwr_4",
            "sky130_fd_sc_hd__lpflow_decapkapwr_6",
            "sky130_fd_sc_hd__lpflow_decapkapwr_8",
            "sky130_fd_sc_hd__lpflow_inputiso0n_1",
            "sky130_fd_sc_hd__lpflow_inputiso0p_1",
            "sky130_fd_sc_hd__lpflow_inputiso1n_1",
            "sky130_fd_sc_hd__lpflow_inputiso1p_1",
            "sky130_fd_sc_hd__lpflow_inputisolatch_1",
            "sky130_fd_sc_hd__lpflow_isobufsrc_1",
            "sky130_fd_sc_hd__lpflow_isobufsrc_16",
            "sky130_fd_sc_hd__lpflow_isobufsrc_2",
            "sky130_fd_sc_hd__lpflow_isobufsrc_4",
            "sky130_fd_sc_hd__lpflow_isobufsrc_8",
            "sky130_fd_sc_hd__lpflow_isobufsrckapwr_16",
            "sky130_fd_sc_hd__lpflow_lsbuf_lh_hl_isowell_tap_1",
            "sky130_fd_sc_hd__lpflow_lsbuf_lh_hl_isowell_tap_2",
            "sky130_fd_sc_hd__lpflow_lsbuf_lh_hl_isowell_tap_4",
            "sky130_fd_sc_hd__lpflow_lsbuf_lh_isowell_4",
            "sky130_fd_sc_hd__lpflow_lsbuf_lh_isowell_tap_1",
            "sky130_fd_sc_hd__lpflow_lsbuf_lh_isowell_tap_2",
            "sky130_fd_sc_hd__lpflow_lsbuf_lh_isowell_tap_4",
            "sky130_fd_sc_hd__sdfbbn_1",
            "sky130_fd_sc_hd__sdfbbn_2",
            "sky130_fd_sc_hd__sdfbbp_1",
            "sky130_fd_sc_hd__sdfrbp_1",
            "sky130_fd_sc_hd__sdfrbp_2",
            "sky130_fd_sc_hd__sdfrtn_1",
            "sky130_fd_sc_hd__sdfrtp_1",
            "sky130_fd_sc_hd__sdfrtp_2",
            "sky130_fd_sc_hd__sdfrtp_4",
            "sky130_fd_sc_hd__sdfsbp_1",
            "sky130_fd_sc_hd__sdfsbp_2",
            "sky130_fd_sc_hd__sdfstp_1",
            "sky130_fd_sc_hd__sdfstp_2",
            "sky130_fd_sc_hd__sdfstp_4",
            "sky130_fd_sc_hd__sdfxbp_1",
            "sky130_fd_sc_hd__sdfxbp_2",
            "sky130_fd_sc_hd__sdfxtp_1",
            "sky130_fd_sc_hd__sdfxtp_2",
            "sky130_fd_sc_hd__sdfxtp_4",
        ]
    
    @property
    def innovus_vars(self) -> list:
        return {
        # floorplan
        'place_site': 'unithd',

        # powerplan
        'pwr_port': '"VPB VPWR"',
        'gnd_port': '"VGND VNB"',
        'stripe_width': 6,
        'stripe_spacing': 2,
        'stripe_distance': 30,
        'stripe_v_layer': 'met4',
        'stripe_h_layer': 'met5',
        'sroute_min_layer': 'li1(1)',
        'sroute_max_layer': 'met4(4)',

        # placement
        'route_min_layer': '2',
        'route_max_layer': '6',

        # cts
        'cts_routing_mul': 2,
        'ndr_cts_min_layer': 'met2',
        'ndr_cts_max_layer': 'met4',
        'cts_inv_cells': ["sky130_fd_sc_hd__clkbuf_4"],
        'cts_buf_cells': [
            "sky130_fd_sc_hd__lpflow_clkinvkapwr_1",
            "sky130_fd_sc_hd__lpflow_clkinvkapwr_2",
            "sky130_fd_sc_hd__lpflow_clkinvkapwr_4",
            "sky130_fd_sc_hd__lpflow_clkinvkapwr_8",
            "sky130_fd_sc_hd__lpflow_clkinvkapwr_16",
        ],
    }

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(self.innovus_vars)
        return d