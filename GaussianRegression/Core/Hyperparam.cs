using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GaussianRegression.Core
{
    public class Hyperparam
    {
        public readonly Type type;                  //Type of this item
        public double value { get; private set; }
        protected Hyperparam(double v, Type t)
        {
            type = t;
            value = v;
        }

        public override int GetHashCode()
        {
            return type.GetHashCode();
        }

        public override bool Equals(object obj)
        {
            if (obj.GetType() != this.GetType())
                return false;
            var o = (Hyperparam)obj;
            if (o.type != this.type || !o.value.Equals(this.value))
                return false;
            return true;
        }


        public static Hyperparam createInstance(Type type, double value)
        {
            if(!type.IsSubclassOf(typeof(Hyperparam)))
                throw new InvalidCastException("Wrong Type Input for Hyperparam: " + type);
            
            var obj = Activator.CreateInstance(type, new Object[1] { value });
            dynamic changedObj = Convert.ChangeType(obj, type);
            return changedObj;
        }
    }

    public sealed class LengthScale : Hyperparam
    {
        public LengthScale(double v) : base(v, typeof(LengthScale)) {
        }
    }

    public sealed class SigmaF : Hyperparam
    {
        public SigmaF(double v) : base(v, typeof(SigmaF))
        {
        }
    }

    public sealed class SigmaJ : Hyperparam
    {
        public SigmaJ(double v) : base(v, typeof(SigmaJ))
        {
        }
    }

    public sealed class Dof : Hyperparam
    {
        public Dof(double v) : base(v, typeof(Dof))
        {
        }
    }
}
