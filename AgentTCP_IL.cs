using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;


public class AgentTCP : Agent
{
    
    Rigidbody rBody;
    
    public FixedJoint AgentJoint;
    public Quaternion initialRotation;
    private Quaternion targetInitialRotation;
    public float minDistance = 100.0f;

    void Start () {
        this.MaxStep = 4000;
        rBody = GetComponent<Rigidbody>();   
        AgentJoint = GetComponent<FixedJoint>();
        initialRotation = this.transform.rotation;
        targetInitialRotation = Target.rotation;
    }

    public Transform Target;
    public Transform Block;
    public Vector3 lastVelocity = Vector3.zero;
    public Vector3 lastAngularVelocity = Vector3.zero;
    public Vector3 obs_force = Vector3.zero;
    public Vector3 obs_torque = Vector3.zero;
    public override void OnEpisodeBegin()
    {
        this.rBody.angularVelocity = Vector3.zero;
        this.rBody.velocity = Vector3.zero;
        this.transform.localPosition = new Vector3( 0.05f, 0.05f, 0f);
        this.transform.localRotation = initialRotation;
        // Move the target to a new spot
        Target.localPosition = new Vector3((Random.value * 0.02f),0.02f,(Random.value*0.02f));
        // Reset the target's rotation to its initial rotation
        Target.rotation = targetInitialRotation;
    }
    private void ControlTimeScale()
    {
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            Time.timeScale = 0.2f;  // Set time scale to 0.5 (half speed)
        }
        else if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            Time.timeScale = 1.0f;  // Set time scale to 1.0 (normal speed)
        }
        else if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            Time.timeScale = 2.0f;  // Set time scale to 2.0 (double speed)
        }
    }
        
    public override void CollectObservations(VectorSensor sensor)
    {
        // observation space 24 (vector3 x 8)
        // Target and Gripped block positions
        sensor.AddObservation(Target.localPosition); // vector 3
        sensor.AddObservation(Target.localRotation.eulerAngles); // vector 3 (degrees)
        sensor.AddObservation(Block.localPosition); // vector 3
        sensor.AddObservation(Block.localRotation.eulerAngles); // vector 3 (degrees)

        // Agent velocity
        obs_force = AgentJoint.currentForce;
        obs_torque = AgentJoint.currentTorque;

        sensor.AddObservation(obs_force);
        sensor.AddObservation(obs_torque);

        lastVelocity = new Vector3(rBody.velocity.x, rBody.velocity.y,rBody.velocity.z);
        lastAngularVelocity = new Vector3(rBody.angularVelocity.x, rBody.angularVelocity.y, rBody.angularVelocity.z);
        
        sensor.AddObservation(rBody.velocity);
        sensor.AddObservation(rBody.angularVelocity);

    }


    private float CalculateAngleReward(float angle)
    {
        // Define your reward parameters
        float maxAngleReward = 50f; // Maximum reward for perfect alignment
        float minAngleThreshold = 20.0f; // Minimum angle for any reward
        float maxAngleThreshold = 90.0f; // Maximum angle for full reward

        // Calculate the reward based on the angle difference
        if (angle < minAngleThreshold) 
        {
            // Reward for being very close
            return maxAngleReward - Mathf.Abs(angle);
        }
        else if (angle == 0) 
        {
            return maxAngleReward;
        }
        else if (angle < maxAngleThreshold) 
        {
            // Linearly interpolate the reward between min and max thresholds
            float t = (angle - minAngleThreshold) / (maxAngleThreshold - minAngleThreshold);
            return maxAngleReward * (1.0f - t);
        }
        else
        {
            // No reward for angles greater than max threshold
            return 0.0f;
        }
    }
    void Update()
    {
        ControlTimeScale();  // Call the method to control time scale
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        int currentEpisode = this.CompletedEpisodes;
        int currentStep = this.StepCount;
        var statsRecorder = Academy.Instance.StatsRecorder;
        
        // Actions, size = 6
        Vector3 controlSignalV = Vector3.zero;
        Vector3 controlSignalA = Vector3.zero;
        controlSignalV.x = actionBuffers.ContinuousActions[0];
        controlSignalV.y = actionBuffers.ContinuousActions[1];
        controlSignalV.z = actionBuffers.ContinuousActions[2];
        controlSignalA.x = actionBuffers.ContinuousActions[3];
        controlSignalA.y = actionBuffers.ContinuousActions[4];
        controlSignalA.z = actionBuffers.ContinuousActions[5];
        rBody.velocity = new Vector3(controlSignalV.x,controlSignalV.y,controlSignalV.z);
        rBody.angularVelocity = new Vector3(controlSignalA.x,controlSignalA.y,controlSignalA.z);

        rBody.velocity.Normalize();
        rBody.velocity *= 0.01f;

        rBody.position += rBody.velocity * Time.deltaTime;
        Quaternion deltaRotation = Quaternion.Euler(rBody.angularVelocity * Time.deltaTime);
        rBody.MoveRotation(rBody.rotation * deltaRotation);

        float distanceToTarget = Vector3.Distance(Block.localPosition, Target.localPosition);

        // Calculate the angle between TCP and target rotations
        float angle = Quaternion.Angle(Block.rotation, Target.rotation);

        if (distanceToTarget < minDistance) 
        {
            minDistance = distanceToTarget;
        }

        // Agent velocity
        float joint_force = AgentJoint.currentForce.magnitude;
        float joint_torque = AgentJoint.currentTorque.magnitude;


        // Tensorboard
        statsRecorder.Add("Distance To Target", distanceToTarget); 
        statsRecorder.Add("Angular distance", angle); 
        statsRecorder.Add("Best Score", minDistance); 

        Debug.Log("dT:"+Time.deltaTime+" Ep:"+currentEpisode+" Step:"+currentStep+" V:"+rBody.velocity+" aV:"+rBody.angularVelocity+" F:"+joint_force+" T:"+joint_torque+" D:"+distanceToTarget+"min D:"+minDistance+" a:"+angle);


        // Reward components

        // Component 1: Distance Reward
        // Encourages the agent to get closer to the target.
        // The reward decreases as the distance increases.
        float epsilon = 0.001f; // Distance threshold
        float largePositiveReward = 100.0f; // Large positive reward
        float distanceReward = Mathf.Abs(distanceToTarget) < epsilon
            ? Mathf.Abs(distanceToTarget) + largePositiveReward
            : (1f - Mathf.Abs(distanceToTarget));

        // Component 2: Alignment Reward
        // Encourages the agent to align with the target.
        float angleReward = CalculateAngleReward(angle); // Calculated using your angle reward function

        // Penalty components

        // Component 3: Force Penalty
        // Penalizes the agent when it exceeds the joint force limit.
        float forcePenalty = joint_force > 300.0f ? -1.0f : 0.0f;

        // Component 4: Torque Penalty
        // Penalizes the agent when it exceeds the joint torque limit.
        float torquePenalty = joint_torque > 40.0f ? -1.0f : 0.0f;

        // Calculate the main reward by combining component

        // The main reward is a combination of the distance reward, alignment reward,
        // force penalty, and torque penalty.
        float reward = distanceReward + angleReward + forcePenalty + torquePenalty;

        // Add the reward to the agent
        AddReward(reward);

        // Terminate episode if certain conditions are met

        // Terminate the episode if the agent is too far from the target, smaller than epsilon,
        // or if it exceeds force and torque limits.
        if (distanceToTarget > 0.05f || distanceToTarget < epsilon || joint_force > 300.0f || joint_torque > 40.0f)
        {
            Debug.Log("Terminating episode.");
            EndEpisode();
        }
    }

        // float epsilon = 0.018f; // Distance threshold
        // float largePositiveReward = 100.0f; // Large positive reward
        // float smallPenalty = -1.0f; // Small penalty for exceeding thresholds

        // if (distanceToTarget > 0.05f)
        // {
        //     Debug.Log("Over 100mm away. End episode");
        //     AddReward(smallPenalty);
        //     EndEpisode();
        // }

        // if (joint_force > 100.0f)
        // {
        //     Debug.Log("joint_force too high. End episode");
        //     AddReward(smallPenalty);
        //     EndEpisode();
        // }

        // if (joint_torque > 20.0f)
        // {
        //     Debug.Log("joint_torque too high. End episode");
        //     AddReward(smallPenalty);
        //     EndEpisode();
        // }
        // //Calculate the main reward based on distance 
        // float reward = Mathf.Abs(distanceToTarget) < epsilon
        //     ? Mathf.Abs(distanceToTarget) + largePositiveReward
        //     : (1f - Mathf.Abs(distanceToTarget));

        // AddReward(reward + angleReward);
    //}

//         // if (distanceToTarget > 0.3f)
//         // {
//         //     Debug.Log("Over 300mm away. End episode");
//         //     // AddReward(-1.0f);
//         //     EndEpisode();
//         // }

//         // if (Block.localPosition.y < 0)
//         // {
//         //     Debug.Log("Below table. End episode");
//         //     // AddReward(-1.0f);
//         //     EndEpisode();
//         // }

//         // if (joint_force > 100.0f)
//         // {            
//         //     Debug.Log("joint_force too high. End episode");
//         //     AddReward(-1.0f);
//         //     EndEpisode();
//         // }

//         // if (joint_torque > 20.0f)
//         // {
//         //     Debug.Log("joint_torque too high. End episode");
//         //     AddReward(-1.0f);
//         //     EndEpisode();            
//         // }

//         // // Reached target
//         // if (distanceToTarget < 0.005f)
//         // {
//         //     SetReward(1.0f);
            
//         //     Debug.Log("Reach target at" + distanceToTarget +"mm");
//         //     Debug.Log("End episode");
//         //     EndEpisode();
//         // }
//         // else if (distanceToTarget < 0.05f)
//         // {
//         //     SetReward(1.0f - distanceToTarget * 20.0f - currentStep * 0.01f);
//         // } 
//         // else
//         // {
//         //     SetReward(0.0f - distanceToTarget * 1.0f - currentStep * 0.01f);
//         // }

    


    public override void Heuristic(in ActionBuffers actionsOut)
    {
        float speedFactor = 0.2f;
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        // apply the action as input (heuristic) 
        if (Input.GetKey(KeyCode.W))
        {
            continuousActions[0] = speedFactor; // Set the X value to go up
        }
        else if (Input.GetKey(KeyCode.S))
        {
            continuousActions[0] = -1 * speedFactor; // Set the X value to go down
        }

        if (Input.GetKey(KeyCode.Space))
        {
            continuousActions[1] = speedFactor; // Set the Y value to go up
        }
        else if (Input.GetKey(KeyCode.LeftControl))
        {
            continuousActions[1] = -1 * speedFactor; // Set the Y value to go down
        }

        if (Input.GetKey(KeyCode.A))
        {
            continuousActions[2] = speedFactor; // Set the Z value to go up
        }
        else if (Input.GetKey(KeyCode.D))
        {
            continuousActions[2] = -1 * speedFactor; // Set the Z value to go down
        }

        if (Input.GetKey(KeyCode.I))
        {
            continuousActions[3] = 1.0f; // X rotation up
        }
        else if (Input.GetKey(KeyCode.K))
        {
            continuousActions[3] = -1 * 1.0f; // X rotation down
        }

        if (Input.GetKey(KeyCode.U))
        {
            continuousActions[4] = 1.0f; // Y rotation up
        }
        else if (Input.GetKey(KeyCode.H))
        {
            continuousActions[4] = -1 * 1.0f; // Y rotation down
        } 

        if (Input.GetKey(KeyCode.J))
        {
            continuousActions[5] = 1.0f; // Z rotation up
        }
        else if (Input.GetKey(KeyCode.L))
        {
            continuousActions[5] = -1 * 1.0f; // Z rotation down
        }
    }

    



}