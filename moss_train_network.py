"""Compatible LyCORIS entry point for the official MOSS-Image network trainer."""

from moss_image_train_network import (
    MossImageNetworkTrainer,
    setup_parser as official_setup_parser,
)
from library import (
    flux2_vae as flux2_vae,
)  # compatibility for callers using the legacy module
import library.args as args_util
from library import moss_image_train_utils


class MossNetworkTrainer(MossImageNetworkTrainer):
    def train(self, args):
        if args.network_train_text_encoder_only or args.moss_train_text_encoder:
            raise ValueError(
                "MOSS LyCORIS presets keep Qwen frozen; use the official entry point for TE adapters"
            )
        if args.in_json is not None or args.dataset_class is not None:
            raise ValueError(
                "Use image directories with TXT/JSON caption sidecars for MOSS"
            )
        if args.network_module != "lycoris.kohya":
            raise ValueError(
                "MOSS uses --network_module lycoris.kohya (algo=lora or lokr)"
            )
        network_args = dict(value.split("=", 1) for value in (args.network_args or []))
        network_args.setdefault("preset", "moss")
        if network_args["preset"] not in ("moss", "moss-full") or network_args.get(
            "algo", "lora"
        ) not in ("lora", "lokr"):
            raise ValueError("Use preset=moss/moss-full and algo=lora/lokr")
        args.network_args = [f"{key}={value}" for key, value in network_args.items()]
        return super().train(args)


def setup_parser():
    parser = official_setup_parser()
    # Preserve the old command's discrete uniform training grid. New official
    # timestep distributions are selected explicitly with --timestep_sampling.
    parser.set_defaults(
        network_module="lycoris.kohya",
        timestep_sampling="legacy",
        moss_sample_steps=30,
        moss_sample_seed=42,
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)
    if args.show_timesteps:
        moss_image_train_utils.show_timesteps(args)
    else:
        MossNetworkTrainer().train(args)
