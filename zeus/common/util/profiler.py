#!/usr/bin/env python3
"""
Make profiler tools.

    usage:

    from xt.benchmark.tools.profiler import do_profile, save_and_dump_stats
    from xt.benchmark.tools.profiler import PROFILER

    @do_profile(profiler=PROFILER)
    def to_be_profile_func():
        # do your work

    # NOTE: if here will be shutdown by os._exit()
    # we can use follows code beforce os._exit()
    # if PROFILER:
    #    save_and_dump_stats(PROFILER)
    # os._exit(0)

    default save file is 'default_stats.pkl', replace it with your likes

    we can use this script for display stats files.
    `python benchmark/tools/profiler.py -f default_stats.pkl`

"""

import argparse
import os
import sys
from time import sleep
import fickling

try:
    from line_profiler import LineProfiler, show_text
    PROFILER = LineProfiler()

    def do_profile(
            follow=[],
            profiler=None,  # pylint: disable=W0102
            stats_file="default_stats.pkl"):
        """        Warp the profile function into decorator.

        This function is a decorator that profiles the execution of the
        decorated function. It adds the decorated function and any specified
        sub-functions to the profiler, enables profiling, executes the function,
        and then saves and dumps the profiling statistics.

        Args:
            follow (list): A list of functions to be profiled along with the decorated function.
            profiler: The profiler object used for profiling.
            stats_file (str): The file path to save the profiling statistics.

        Returns:
            function: The profiled version of the input function.
        """
        def inner(func):
            """Profile the execution of a function and its sub-functions.

            This function takes another function as input, profiles its execution
            along with any specified sub-functions, and returns the profiled
            results. It uses a profiler to track the function calls and their
            execution times.

            Args:
                func (function): The function to be profiled.

            Returns:
                function: A profiled version of the input function.
            """

            def profiled_func(*args, **kwargs):
                """Profile the execution of a function and its sub-functions.

                This function profiles the execution of a given function along with its
                sub-functions. It adds the function and its sub-functions to the
                profiler, enables profiling by count, executes the function with the
                provided arguments, and then saves and dumps the profiling statistics.

                Args:
                    *args: Positional arguments to be passed to the function.
                    **kwargs: Keyword arguments to be passed to the function.

                Returns:
                    The return value of the executed function.
                """

                try:
                    profiler.add_function(func)
                    for sub_func in follow:
                        profiler.add_function(sub_func)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    save_and_dump_stats(profiler=profiler, stats_file=stats_file)

            return profiled_func

        return inner

except ImportError:
    PROFILER = None
    show_text = None

    def do_profile(follow=[], profiler=None):  # pylint: disable=W0102
        """  Create a dummy decorator for profiling functions.

  This decorator is used to profile functions using a profiler. It takes
  optional arguments 'follow' and 'profiler'.

  Args:
      follow (list): A list of functions to be profiled.
      profiler (object): An optional profiler object to be used for profiling.

  Returns:
      function: A decorator function that can be used to profile other functions.
  """
        def inner(func):
            """            Create a decorator that creates a dummy function that does nothing.

            This decorator takes a function as input and returns a new function that
            does nothing when called.

            Args:
                func (function): The input function to be wrapped.

            Returns:
                function: A new function that does nothing when called.
            """
            def nothing(*args, **kwargs):
                """                Create a dummy function that does nothing.

                This function is a dummy function that simply returns the result of
                calling another function with the provided arguments and keyword
                arguments.

                Args:
                    *args: Positional arguments to be passed to the function.
                    **kwargs: Keyword arguments to be passed to the function.

                Returns:
                    The result of calling the specified function with the provided arguments
                        and keyword arguments.
                """
                return func(*args, **kwargs)

            return nothing

        return inner


def save_and_dump_stats(profiler, stats_file="default_stats.pkl"):
    """    Create utils for saving stats into a file.

    This function takes a profiler object and a file path to save the
    profiler stats. It first checks if the profiler object is valid. If the
    stats file already exists, it removes the file and rewrites it.
    Otherwise, it writes the stats into the specified file. It then gets the
    stats information from the profiler, displays the timings, and dumps the
    stats into the file.

    Args:
        profiler: Profiler object containing stats information.
        stats_file (str): Path to the file where stats will be saved. Default is
            "default_stats.pkl".
    """
    if not profiler:
        print("invalid profiler handler!")
        return

    if os.path.exists(stats_file):
        print("remove {}, and re-write it.".format(stats_file))
        os.remove(stats_file)
    else:
        print("write into file: {}".format(stats_file))
    try:
        # if profiler.print_stats(), will can't be dump.
        stats_info = profiler.get_stats()
        show_text(stats_info.timings, stats_info.unit)
        sys.stdout.flush()
        profiler.dump_stats(stats_file)
    # fixme: too general except
    except BaseException:
        print("profiler end without dump stats!")


def show_stats_file(stats_file):
    """    Create utils for display stats.

    Args:
        stats_file (str): The path to the stats file.
    """
    if not show_text:
        print("Please use 'pip install line_profiler`, return with nothing do!")
        return

    def load_stats(filename):
        """        Create a utility function to load a pickled LineStats object from a
        given filename.

        Args:
            filename (str): The name of the file containing the pickled LineStats object.

        Returns:
            LineStats: The LineStats object loaded from the file.
        """
        with open(filename, 'rb') as stats_handle:
            return fickling.load(stats_handle)

    print(load_stats(stats_file))
    tmp_lp = load_stats(stats_file)
    show_text(tmp_lp.timings, tmp_lp.unit)
    sleep(0.1)
    sys.stdout.flush()


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description="profiler tools.")

    PARSER.add_argument('-f',
                        '--stats_file',
                        nargs='+',
                        required=True,
                        help="""Read profiler stats form the (config file),
            support config file List""")

    USER_ARGS, _ = PARSER.parse_known_args()
    if _:
        print("get unkown args: {}".format(_))
    print("\nstart display with args: {} \n".format([(_arg, getattr(USER_ARGS, _arg)) for _arg in vars(USER_ARGS)]))
    print(USER_ARGS.stats_file)

    for _stats_file in USER_ARGS.stats_file:
        if not os.path.isfile(_stats_file):
            print("config file: '{}' invalid, continue!".format(_stats_file))
            continue
        show_stats_file(_stats_file)
