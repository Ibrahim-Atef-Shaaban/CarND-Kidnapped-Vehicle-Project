/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;
  num_particles = 100;  // TODO: Set the number of particles
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  particles.resize(num_particles);
  
  for(auto& particle: particles){
    particle.id = &particle - &particles[0];
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  std::normal_distribution<double> noise_x(0, std_pos[0]);
  std::normal_distribution<double> noise_y(0, std_pos[1]);
  std::normal_distribution<double> noise_theta(0, std_pos[2]);
  
  for(auto& particle: particles){
    if (fabs(yaw_rate) < 0.0001){
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.x += velocity * delta_t * sin(particle.theta);
    }
    else{
    particle.x += velocity/yaw_rate * (sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta));
    particle.y += velocity/yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t));
    particle.theta += yaw_rate * delta_t;
    }
    particle.x += noise_x(gen);
    particle.y += noise_y(gen);
    particle.theta += noise_theta(gen);
  }
  
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(auto& observation: observations){
    double minDist = std::numeric_limits<double>::max();
    int minID = std::numeric_limits<int>::min();
    for(auto& prediction: predicted){
      double Dist = dist(observation.x, observation.y, prediction.x, prediction.y);
      if (Dist < minDist)
      {
        minDist = Dist;
        minID = prediction.id;
      }
    }
    observation.id = minID;
  }
    

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for(auto& particle: particles){
    particle.weight = 1;
    // Eliminate invalid landmarks
    vector<LandmarkObs> predictions;
    for(auto& landmark: map_landmarks.landmark_list){
      double distance = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
      if( distance < sensor_range){
        predictions.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }
    // Change observed objects from vehicle coordinates to map coordinates
    vector<LandmarkObs> observations_map;
    for(auto& observation: observations){
      LandmarkObs temp_obs;
      temp_obs.id = observation.id;
      temp_obs.x = observation.x * cos(particle.theta) - observation.y * sin(particle.theta) + particle.x;
      temp_obs.y = observation.x * sin(particle.theta) + observation.y * cos(particle.theta) + particle.y;
      observations_map.push_back(temp_obs);
    }
    // Match the Predections and the transformed observations
    dataAssociation(predictions, observations_map);
    for(auto& observation: observations_map){
      Map::single_landmark_s landmarkmatch = map_landmarks.landmark_list.at(observation.id -1);
      double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      double exponent = (pow(observation.x - landmarkmatch.x_f, 2) / (2 * pow(std_landmark[0], 2)))
        				+ (pow(observation.y - landmarkmatch.y_f, 2) / (2 * pow(std_landmark[1], 2)));
      double weight = gauss_norm * exp(-exponent);
      particle.weight *= weight;
    }
    weights.push_back(particle.weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> distribution(weights.begin(), weights.end());
  
  vector<Particle> resampled_particles;
  resampled_particles.resize(num_particles);
    
  for (int i =0; i<num_particles; i++){
    int gen_idx = distribution(gen);
    resampled_particles[i] = particles[gen_idx];
  }
  particles = resampled_particles;
  weights.clear();

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}