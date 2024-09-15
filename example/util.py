from concurrent.futures import Future

from cassandra.cluster import EXEC_PROFILE_DEFAULT
from cassandra.concurrent import ConcurrentExecutorListResults

class ConcurrentExecutorFutureResults(ConcurrentExecutorListResults):
    def __init__(self, session, statements_and_params, execution_profile, future):
        super().__init__(session, statements_and_params, execution_profile)
        self.future = future

    def _put_result(self, result, idx, success):
        super()._put_result(result, idx, success)
        with self._condition:
            if self._current == self._exec_count:
                if self._exception and self._fail_fast:
                    self.future.set_exception(self._exception)
                else:
                    sorted_results = [r[1] for r in sorted(self._results_queue)]
                    self.future.set_result(sorted_results)


def execute_concurrent_async(
    session,
    statements_and_parameters,
    concurrency=100,
    raise_on_first_error=False,
    execution_profile=EXEC_PROFILE_DEFAULT
):
    """
    Asynchronously executes a sequence of (statement, parameters) tuples concurrently.

    Args:
        session: Cassandra session object.
        statement_and_parameters: Iterable of (prepared CQL statement, bind parameters) tuples.
        concurrency (int, optional): Number of concurrent operations. Default is 100.
        raise_on_first_error (bool, optional): If True, execution stops on the first error. Default is True.
        execution_profile (ExecutionProfile, optional): Execution profile to use. Default is EXEC_PROFILE_DEFAULT.

    Returns:
        A `Future` object that will be completed when all operations are done.
    """
    # Create a Future object and initialize the custom ConcurrentExecutor with the Future
    future = Future()
    executor = ConcurrentExecutorFutureResults(
        session=session,
        statements_and_params=statements_and_parameters,
        execution_profile=execution_profile,
        future=future
    )

    # Execute concurrently
    try:
        executor.execute(concurrency=concurrency, fail_fast=raise_on_first_error)
    except Exception as e:
        future.set_exception(e)

    return future
