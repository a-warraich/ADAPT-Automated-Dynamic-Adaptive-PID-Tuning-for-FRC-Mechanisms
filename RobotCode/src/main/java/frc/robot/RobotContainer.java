package frc.robot;

import frc.robot.subsystems.GAPIDSparkMaxSubsystem;

public class RobotContainer {

    public final GAPIDSparkMaxSubsystem gaPidSubsystem = new GAPIDSparkMaxSubsystem();

    public RobotContainer() {
        configureBindings();
    }

    private void configureBindings() {
    }
}
