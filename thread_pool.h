#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

class thread_pool
{
public:
  thread_pool(size_t);
  thread_pool();
  void init(size_t);
  template<typename F, typename... Args>
  auto enqueue(F&& f, Args&&... args)->std::future < typename std::result_of<F(Args...)>::type > ;
  void close();
  ~thread_pool();

private:
  // need to keep track of threads so we can join them
  std::vector< std::thread > workers;
  // the task queue
  std::queue<std::function<void()>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

// the constructor just launches some amount of workers
inline thread_pool::thread_pool(size_t threads)
  : stop(true)
{
  init(threads);
}

inline thread_pool::thread_pool()
  : stop(true)
{}

inline void thread_pool::init(size_t threads)
{
  if (!stop)
    return;
  stop = false;
  for (size_t i = 0; i < threads; ++i)
  {
    workers.emplace_back(
      [this]
    {
      for (;;)
      {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->queue_mutex);
          this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
          if (this->stop && this->tasks.empty())
            return;
          task = std::move(this->tasks.front());
          this->tasks.pop();
        }
        task();
      }
    });
  }
}

// add new work item to the pool
template<typename F, typename... Args>
auto thread_pool::enqueue(F&& f, Args&&... args) -> std::future < typename std::result_of<F(Args...)>::type >
{
  if (stop)
    throw std::runtime_error("enqueue on stoped thread_pool");
  if (workers.empty())
    throw std::runtime_error("enqueue on no worker thread_pool");

  using return_type = typename std::result_of<F(Args...)>::type ;
  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    if (stop)
      throw std::runtime_error("enqueue on stoped thread_pool");

    tasks.emplace([task](){(*task)(); });
  }
  condition.notify_one();
  return res;
}

inline void thread_pool::close()
{
  if (stop)
    return;
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread &worker : workers)
    worker.join();
  workers.clear();
}

// the destructor joins all threads
inline thread_pool::~thread_pool()
{
  close();
}