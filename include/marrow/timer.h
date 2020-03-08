/*
 * Copyright 2014 MarrowTeam
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
 *  March 2014: Herve Paulino
 *      Specification and implementation of the timer class (from pre-existent classes)
 *
 */

#ifndef MARROW_UTILS_TIMER_HPP
#define MARROW_UTILS_TIMER_HPP

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <iostream>


namespace marrow {

    static const std::string main_stage = "_";

    template <class Duration = std::chrono::milliseconds>
    class timer {

        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = std::chrono::time_point<Clock>;
        using ElapseTime = typename Duration::rep;


        struct stage {

            /**
             * Measurements
             */
            std::vector<ElapseTime> measurements;

            /**
             * Start of next measurement
             */
            TimePoint start;

            stage() :
                    measurements (),
                    start (std::forward<TimePoint>(Clock::now()))
            {}


            // Auxiliary variables for computing averages and standard deviations
            unsigned account_from;
            unsigned account_to;
            unsigned number_measurements;
        };


		public:

        /**
         *
         * @param percentage Percentage of the middle measurements to use in the statistics.
         * For instance, for percentage = 90, the lower and upper 5% and not considered in the statistics.
         * Values <= 0 or > 100 are ignored.
         */
        timer(const unsigned char percentage = 100) :
            percentage (percentage > 100 ? 100 : percentage)
        { }

        /**
         * Start a new measurement
         * @param stage_name Name of the stage: the default is main stage denoted by "_"
         */
        void start(const std::string& stage_name = main_stage) {
            auto s = stages.find(stage_name);
            if (s == stages.end())
                stages.emplace(stage_name, stage{});
            else
              s->second.start = Clock::now();
        }

        /**
         * Stop a new measurement
         * @param stage_name Name of the stage: the default is main stage denoted by "_"
         * @return The time elapsed from the last start
         */
        ElapseTime stop(const std::string& stage_name = main_stage) {
            auto now = Clock::now();

            timer::stage &s = stages.at(stage_name);
            auto elapsed = std::chrono::duration_cast<Duration>(now - s.start).count();
            s.measurements.push_back(elapsed);

            return elapsed;
        }

        /**
         * Reset the measurements for a given stage
         * @param stage_name Name of the stage: the default is main stage denoted by "_"
         */
        void reset(const std::string& stage_name = main_stage) {
            timer::stage &s = stages.at(stage_name);
            s.measurements.clear();
        }

        /**
         * Obtain the average of the measurements for a given stage
         * @param stage_name Name of the stage: the default is main stage denoted by "_"
         * @return The average
         */
        ElapseTime average(const std::string& stage_name = main_stage) {
            stage &s = stages.at(stage_name);

            auto n_measurements = s.measurements.size();
            if (n_measurements <= 1)
                return s.measurements[0];

            qsort(s.measurements.data(), n_measurements, sizeof(ElapseTime), compare);

            if (percentage < 100) {
                s.number_measurements = n_measurements - round(n_measurements * percentage / 100);
                s.account_from = s.number_measurements == n_measurements ?
                               0 :
                               round((n_measurements - s.number_measurements) / 2 - 1);
                s.account_to = s.account_from + s.number_measurements;
            } else {
                s.number_measurements = n_measurements;
                s.account_from = 0;
                s.account_to = s.number_measurements;
            }

            ElapseTime average = 0;
            for (unsigned i = s.account_from; i < s.account_to; i++)
                average += s.measurements[i];
            average /= s.number_measurements;

            return average;
        }

        /**
         * Obtain the standard deviation of the measurements for a given stage
         * @param stage_name Name of the stage: the default is main stage denoted by "_"
         * @return The standard deviation
         */
        ElapseTime std_deviation(const std::string& stage_name = main_stage) {
            stage &s = stages.at(stage_name);
            const auto avg = average(stage_name);

            if (s.number_measurements <= 1)
                return 0;

            double variance = 0.0;
            for (unsigned int i = s.account_from; i <= s.account_to; i++) {
                auto aux = s.measurements[i] - avg;
                variance += aux * aux;
            }
            variance /= s.number_measurements;

            return std::sqrt(variance);
        }

        /**
         * Output the statistics of a given stage for the supplied output stream
         * @param out The output stream
         * @param stage_name Name of the stage: the default is main stage denoted by "_"
         * @param cvs Output in CVS format?
         */
        void output_stats(std::ostream& out,
                         const std::string& stage_name = main_stage,
                         const bool cvs = true) {

            stage &s = stages.at(stage_name);

            if (s.number_measurements <= 1)
                out << s.measurements[0];

            else if (cvs) {
                out << s.number_measurements << "/" << s.measurements.size() <<

                    "," <<
                    " & " << s.measurements[s.account_from] <<
                    " & " << s.measurements[s.account_to] <<
                    " & " << average(stage_name) <<
                    " & " << std_deviation(stage_name);
            } else {
                out << "statistics (middle " << s.number_measurements << " of " << s.measurements.size() <<
                    " measurements) in " << ":" << std::endl <<
                    "\tAverage: " << average(stage_name) << std::endl <<
                    "\tMaximum: " << s.measurements[s.account_to] << std::endl <<
                    "\tMinimum: " << s.measurements[s.account_from] << std::endl <<
                    "\tStandard deviation: " << std_deviation(stage_name) << std::endl;
            }
        }

    private:

        /**
         * Map of stage identifiers into stages
         */
        std::map<std::string, stage> stages;

        /**
         * Percentage of the middle measurements to use in the statistics
         */
        const unsigned char percentage;

        /**
         * Auxiliary compare function
         */
        static int compare(const void *a, const void *b) {
            const ElapseTime ta = * (const ElapseTime *) a;
            const ElapseTime tb = * ((const ElapseTime *) b);

            return (ta < tb) ? 1 : ta > tb;
        }
    };
}

#endif// MARROW_UTILS_TIMER_HPP


